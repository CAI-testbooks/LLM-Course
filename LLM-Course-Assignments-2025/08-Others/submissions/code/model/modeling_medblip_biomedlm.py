import logging
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from lavis.models.blip2_models.blip2 import Blip2Base
from model.modeling_gpt2 import GPT2LMHeadModel
from model.eva_vit import create_eva_vit_g
from transformers import GPT2Tokenizer

import random
from data.gen_brain_qa import generate_adhd_qa_txt, generate_hcp_txt
from types import SimpleNamespace

logger = logging.getLogger(__name__)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class MedBLIPModel_biomedlm(Blip2Base):
    """
    MedBLIP variant integrating:
      - learnable prompt tokens
      - multi-layer visual->text alignment with optional adapters per layer
      - adapter config: use_adapter, adapter_dim, adapter_dropout
      - layer-wise alignment weights (static list or learnable)
      - adapter projection into LM embedding space for use during generation
    """
    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        patch_size=32,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        lm_model="stanford-crfm/BioMedLM",
        prompt="",
        max_txt_len=256,
        apply_lemmatizer=False,
        embed_dim=256,
        num_prompt_tokens=32,
        align_layer_idxs=None,
        align_loss_weight=1.0,
        # ---------- adapter & align weights ----------
        use_adapter: bool = True,
        adapter_dim: int = 32,
        adapter_dropout: float = 0.0,
        align_layer_weights: list = None,   # e.g., [0.2,0.2,0.3,0.3] or None
        align_weights_learnable: bool = False,  # if True, create learnable weights (softmax normalized)
    ):
        super().__init__()

        # -------------- vision encoder --------------
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, patch_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                if '3d' not in name:
                    param.requires_grad = False

        # -------------- Qformer --------------
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        # remove cls head if exists (we only use bert encoder)
        self.Qformer.cls = None

        # -------------- tokenizer & LM --------------
        self.tokenizer = GPT2Tokenizer.from_pretrained(lm_model, pad_token='<PAD>')
        # load LM in fp16 to save memory (like original)
        self.lm_model = GPT2LMHeadModel.from_pretrained(lm_model, torch_dtype=torch.float16)
        for name, param in self.lm_model.named_parameters():
            param.requires_grad = False

        # projections
        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.proj = nn.Linear(self.Qformer.config.hidden_size, self.lm_model.config.n_embd)  # qformer->lm embedding

        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.max_txt_len = max_txt_len

        # ---------------- 新增: 可学习 prompt tokens ----------------
        self.num_prompt_tokens = num_prompt_tokens
        self.prompt_embeddings = nn.Parameter(
            torch.randn(1, self.num_prompt_tokens, self.Qformer.config.hidden_size) * 0.02,
            requires_grad=True
        )

        # -------------- alignment layer config --------------
        vis_feat_dim = self.visual_encoder.num_features  # e.g. 1408
        self.align_loss_weight = align_loss_weight
        # if not specified, pick some defaults (1-based indices)
        if align_layer_idxs is None:
            last_layer_idx = len(self.visual_encoder.blocks)
            self.align_layer_idxs = [4, 8, 12, 16, last_layer_idx]
        else:
            self.align_layer_idxs = align_layer_idxs

        # -------------- per-layer visual -> embed projections --------------
        self.vis_layer_projs = nn.ModuleDict()
        for li in self.align_layer_idxs:
            self.vis_layer_projs[str(li)] = nn.Linear(vis_feat_dim, embed_dim)

        # -------------- adapter settings --------------
        self.use_adapter = use_adapter
        self.adapter_dim = adapter_dim
        self.adapter_dropout = float(adapter_dropout) if adapter_dropout is not None else 0.0

        # create adapters if enabled
        self.vis_layer_adapters = nn.ModuleDict()
        if self.use_adapter:
            for li in self.align_layer_idxs:
                # adapter: embed_dim -> adapter_dim -> ReLU -> Dropout -> adapter_dim -> embed_dim
                self.vis_layer_adapters[str(li)] = nn.Sequential(
                    nn.Linear(embed_dim, self.adapter_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(self.adapter_dropout),
                    nn.Linear(self.adapter_dim, embed_dim),
                )
            # projection from embed_dim to LM embedding dim for adapter outputs (used optionally in generate)
            self.adapter_proj_to_lm = nn.Linear(embed_dim, self.lm_model.config.n_embd)

        # -------------- layer-wise weights (static or learnable) --------------
        self.align_weights_learnable = align_weights_learnable
        n_layers = len(self.align_layer_idxs)
        if self.align_weights_learnable:
            # initialize learnable weights (logits); will be softmaxed when used
            init_logits = torch.ones(n_layers, dtype=torch.float32)
            self.align_layer_weights_param = nn.Parameter(init_logits)
        else:
            # static list or equal weights
            if align_layer_weights is None:
                # equal weights
                self._static_align_layer_weights = [1.0] * n_layers
            else:
                # if user provided fewer/more weights, attempt to broadcast / trim
                if len(align_layer_weights) != n_layers:
                    logger.warning("align_layer_weights length (%d) != number of align_layer_idxs (%d). "
                                   "Will broadcast/trim to match." % (len(align_layer_weights), n_layers))
                # build weights list of length n_layers
                weights = []
                for i in range(n_layers):
                    if i < len(align_layer_weights):
                        weights.append(float(align_layer_weights[i]))
                    else:
                        weights.append(1.0)
                self._static_align_layer_weights = weights

    def init_vision_encoder(
        self,
        model_name,
        img_size,
        patch_size,
        drop_path_rate,
        use_grad_checkpoint,
        precision
    ):
        visual_encoder = create_eva_vit_g(
            img_size,
            patch_size,
            drop_path_rate,
            use_grad_checkpoint,
            precision
        )

        ln_vision = LayerNorm(visual_encoder.num_features)
        return visual_encoder, ln_vision

    def forward(self, samples):
        """
        samples: dict with keys:
            - images: tensor
            - reports: list[str]
            - tag: list[str] (dataset type per sample)
        returns: dict with loss components
        """
        image = samples["images"].cuda().half()
        base_text = []
        question = []
        answer = []
        tq = []
        dataset_type = ''

        bs = len(samples['reports'])
        for b in range(bs):
            report = samples['reports'][b]
            tag = samples['tag'][b]

            if tag == 'adhd':
                doc = generate_adhd_qa_txt(report)
            else:
                doc = generate_hcp_txt(report)

            dataset_type = tag
            base_text.append(doc['base_text'])
            question.append(doc['question'])
            answer.append(doc['answer'])
            tq.append(doc['tq'])

        with self.maybe_autocast():
            # 请求视觉编码器返回中间层 outputs
            final_vis_out = self.visual_encoder(image, return_layer_idxs=self.align_layer_idxs)
            if isinstance(final_vis_out, tuple):
                image_embeds_raw, vis_layer_outputs = final_vis_out
            else:
                image_embeds_raw = final_vis_out
                vis_layer_outputs = None

            image_embeds = self.ln_vision(image_embeds_raw)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        # ---------------- base_text -> Qformer (with prompt embeddings concat) ----------------
        base_text_tokens = self.tokenizer(
            base_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)

        # 尝试用 Qformer 的 word_embeddings，否则回退到 lm_model 的 wte
        try:
            base_input_embeddings = self.Qformer.bert.embeddings.word_embeddings(base_text_tokens.input_ids)
        except Exception:
            base_input_embeddings = self.lm_model.transformer.wte(base_text_tokens.input_ids)

        bs_local = base_input_embeddings.size(0)
        prompt_expand = self.prompt_embeddings.expand(bs_local, -1, -1).to(
            device=base_input_embeddings.device,
            dtype=base_input_embeddings.dtype
        )
        inputs_embeds_for_text = torch.cat([base_input_embeddings, prompt_expand], dim=1)
        prompt_mask = torch.ones((bs_local, self.num_prompt_tokens), dtype=torch.long).to(
            base_text_tokens.attention_mask.device)
        attention_mask_for_text = torch.cat([base_text_tokens.attention_mask, prompt_mask], dim=1)

        # call Qformer.bert robustly
        try:
            base_text_output = self.Qformer.bert(
                inputs_embeds=inputs_embeds_for_text,
                attention_mask=attention_mask_for_text,
                return_dict=True,
            )
        except TypeError:
            bert_module = self.Qformer.bert
            try:
                embedding_output = bert_module.embeddings(inputs_embeds=inputs_embeds_for_text)
            except Exception:
                # fallback
                embedding_output = inputs_embeds_for_text

            attn_mask = attention_mask_for_text
            try:
                encoder_outputs = bert_module.encoder(embedding_output, attention_mask=attn_mask, return_dict=True)
                sequence_output = encoder_outputs.last_hidden_state if hasattr(encoder_outputs, 'last_hidden_state') else encoder_outputs[0]
            except Exception:
                if hasattr(bert_module, 'get_extended_attention_mask'):
                    try:
                        extended_mask = bert_module.get_extended_attention_mask(attn_mask, embedding_output.shape[:-1], embedding_output.device)
                    except TypeError:
                        try:
                            extended_mask = bert_module.get_extended_attention_mask(attn_mask, embedding_output.shape[:-1], embedding_output.device, False)
                        except Exception:
                            extended_mask = attn_mask[:, None, None, :].to(dtype=next(self.parameters()).dtype)
                            extended_mask = (1.0 - extended_mask) * -10000.0
                else:
                    extended_mask = attn_mask[:, None, None, :].to(dtype=next(self.parameters()).dtype)
                    extended_mask = (1.0 - extended_mask) * -10000.0

                try:
                    encoder_outputs = bert_module.encoder(embedding_output, attention_mask=extended_mask, return_dict=True)
                    sequence_output = encoder_outputs.last_hidden_state if hasattr(encoder_outputs, 'last_hidden_state') else encoder_outputs[0]
                except TypeError:
                    encoder_outputs = bert_module.encoder(embedding_output, attention_mask=attn_mask, return_dict=True)
                    sequence_output = encoder_outputs.last_hidden_state if hasattr(encoder_outputs, 'last_hidden_state') else encoder_outputs[0]

            base_text_output = SimpleNamespace(last_hidden_state=sequence_output)

        # 文本全局表征（CLS）
        base_text_feat = F.normalize(self.text_proj(base_text_output.last_hidden_state[:, 0, :]), dim=-1)

        # ---------- LM input construction (image queries mapped to lm emb) ----------
        img_embeds = self.proj(query_output.last_hidden_state)  # bs num_queries lm_n_embd
        atts_img = torch.ones(img_embeds.size()[:-1], dtype=torch.long).to(image.device)

        self.tokenizer.padding_side = "right"
        input_tokens = self.tokenizer(
            tq,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        output_tokens = self.tokenizer(
            answer,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)

        input_targets = input_tokens.input_ids.masked_fill(
            input_tokens.input_ids == self.tokenizer.pad_token_id, -100)
        output_targets = output_tokens.input_ids.masked_fill(
            output_tokens.input_ids == self.tokenizer.pad_token_id, -100)

        empty_img_targets = (
            torch.ones(atts_img.size(), dtype=torch.long).to(image.device).fill_(-100)
        )
        targets = torch.cat([input_targets, empty_img_targets, output_targets], dim=1)

        inputs_txt_embeds = self.lm_model.transformer.wte(input_tokens.input_ids)
        outputs_txt_embeds = self.lm_model.transformer.wte(output_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_txt_embeds, img_embeds, outputs_txt_embeds], dim=1)
        attention_mask = torch.cat([input_tokens.attention_mask, atts_img, output_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.lm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        loss_lm = outputs.loss

        # ---------- ITC loss (original) ----------
        image_feats = F.normalize(self.vision_proj(query_output.last_hidden_state), dim=-1)
        bs_idx = image_feats.size(0)
        targets_itc = torch.arange(bs_idx, dtype=torch.long).to(image.device)

        sim_q2t = torch.matmul(image_feats.unsqueeze(1), base_text_feat.unsqueeze(-1)).squeeze()
        sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp

        sim_t2q = torch.matmul(base_text_feat.unsqueeze(1).unsqueeze(1), image_feats.permute(0, 2, 1)).squeeze()
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp

        loss_itc = (
            F.cross_entropy(sim_i2t, targets_itc, label_smoothing=0.1)
            + F.cross_entropy(sim_t2i, targets_itc, label_smoothing=0.1)
        ) / 2

        # ---------- 新增：逐层视觉->文本对齐 loss（包含 adapter/residual & layer-wise weights） ----------
        align_loss_total = torch.tensor(0.0).to(image.device)
        if vis_layer_outputs is not None:
            # compute weights
            n_layers = len(self.align_layer_idxs)
            if self.align_weights_learnable:
                # softmax normalize learnable logits to get non-negative weights summing to 1
                weight_vals = F.softmax(self.align_layer_weights_param, dim=0).to(image.device)
            else:
                # static list; normalize to sum=1
                weights = torch.tensor(self._static_align_layer_weights, dtype=torch.float32, device=image.device)
                if weights.sum() != 0:
                    weight_vals = weights / weights.sum()
                else:
                    weight_vals = torch.ones_like(weights) / float(n_layers)

            # vis_layer_outputs 与 self.align_layer_idxs 一一对应
            for li_idx, (idx, layer_out) in enumerate(zip(self.align_layer_idxs, vis_layer_outputs)):
                # layer_out shape: bs, seq_len, vis_feat_dim
                layer_cls = layer_out[:, 0, :]  # take CLS token
                proj = self.vis_layer_projs[str(idx)](layer_cls)  # bs, embed_dim

                # adapter + residual (如果开启)
                if self.use_adapter:
                    adapter_module = self.vis_layer_adapters[str(idx)] if str(idx) in self.vis_layer_adapters else None
                    if adapter_module is not None:
                        adapter_out = adapter_module(proj)
                        proj = proj + adapter_out  # residual

                proj = F.normalize(proj, dim=-1)

                sim_matrix = torch.matmul(proj, base_text_feat.t()) / self.temp
                loss_i2t = F.cross_entropy(sim_matrix, targets_itc)
                loss_t2i = F.cross_entropy(sim_matrix.t(), targets_itc)
                layer_loss = 0.5 * (loss_i2t + loss_t2i)

                # apply layer-wise weight
                w = weight_vals[li_idx] if weight_vals.numel() > li_idx else torch.tensor(1.0, device=image.device)
                align_loss_total = align_loss_total + (layer_loss * w)

            # multiply global align weight
            align_loss_total = align_loss_total * self.align_loss_weight

        loss = loss_itc + loss_lm + align_loss_total

        # 返回更多项以便 debug
        # detach scalar components for logging
        return {
            "loss": loss,
            "loss_itc": loss_itc.detach() if isinstance(loss_itc, torch.Tensor) else loss_itc,
            "loss_lm": loss_lm.detach() if isinstance(loss_lm, torch.Tensor) else loss_lm,
            "loss_align": align_loss_total.detach() if isinstance(align_loss_total, torch.Tensor) else align_loss_total
        }

    @torch.no_grad()
    def generate(
        self,
        samples,
        device='cuda:0',
    ):
        """
        Generate with optional adapter-inserted tokens computed from visual intermediate layers.
        If use_adapter=True and visual encoder returns intermediate layers, we compute adapter tokens
        per align layer, project them to LM embedding dim and insert into LM inputs:
           [text_embeds] + [prompt_proj_lm] + [adapter_proj_lm] + [image_proj]
        """
        # 1) Tokenize LM input tokens (these are tokens LM will see as 'input' before image)
        input_tokens = self.tokenizer(
            samples["prompt"],
            padding="longest",
            truncation=True,
            return_tensors="pt",
            max_length=self.max_txt_len
        ).to(device)

        # 2) Prepare images and compute image_embeds (request intermediate layers)
        image = samples["images"].to(device)
        with self.maybe_autocast():
            # request intermediate layers for adapter computation
            image_vis_out = self.visual_encoder(image, return_layer_idxs=self.align_layer_idxs)
            if isinstance(image_vis_out, tuple):
                image_embeds_raw, vis_layer_outputs = image_vis_out
            else:
                image_embeds_raw = image_vis_out
                vis_layer_outputs = None

            image_embeds = self.ln_vision(image_embeds_raw)

        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

        # 3) Qformer query path -> image tokens for LM (unchanged)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1).to(device)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        inputs_img = self.proj(query_output.last_hidden_state)  # (bs, num_queries, lm_n_embd)
        atts_img = torch.ones(inputs_img.size()[:-1], dtype=torch.long).to(device)

        # 4) Qformer text inputs (construct inputs_embeds_for_text = qformer_word_embeds(input_ids) + prompt)
        base_text_tokens = self.tokenizer(
            samples["prompt"],
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)

        try:
            base_input_embeddings = self.Qformer.bert.embeddings.word_embeddings(base_text_tokens.input_ids)
        except Exception:
            base_input_embeddings = self.lm_model.transformer.wte(base_text_tokens.input_ids)

        bs_local = base_input_embeddings.size(0)
        prompt_expand = self.prompt_embeddings.expand(bs_local, -1, -1).to(
            device=base_input_embeddings.device,
            dtype=base_input_embeddings.dtype
        )
        inputs_embeds_for_text = torch.cat([base_input_embeddings, prompt_expand], dim=1)
        prompt_mask = torch.ones((bs_local, self.num_prompt_tokens), dtype=torch.long).to(base_text_tokens.attention_mask.device)
        attention_mask_for_text = torch.cat([base_text_tokens.attention_mask, prompt_mask], dim=1).to(base_input_embeddings.device)

        # 5) Run Qformer on inputs_embeds_for_text robustly (like forward)
        try:
            base_text_output = self.Qformer.bert(
                inputs_embeds=inputs_embeds_for_text,
                attention_mask=attention_mask_for_text,
                return_dict=True,
            )
        except TypeError:
            bert_module = self.Qformer.bert
            try:
                embedding_output = bert_module.embeddings(inputs_embeds=inputs_embeds_for_text)
            except Exception:
                embedding_output = inputs_embeds_for_text

            attn_mask = attention_mask_for_text
            try:
                encoder_outputs = bert_module.encoder(embedding_output, attention_mask=attn_mask, return_dict=True)
                sequence_output = encoder_outputs.last_hidden_state if hasattr(encoder_outputs, 'last_hidden_state') else encoder_outputs[0]
            except Exception:
                if hasattr(bert_module, 'get_extended_attention_mask'):
                    try:
                        extended_mask = bert_module.get_extended_attention_mask(attn_mask, embedding_output.shape[:-1], embedding_output.device)
                    except TypeError:
                        try:
                            extended_mask = bert_module.get_extended_attention_mask(attn_mask, embedding_output.shape[:-1], embedding_output.device, False)
                        except Exception:
                            extended_mask = attn_mask[:, None, None, :].to(dtype=next(self.parameters()).dtype)
                            extended_mask = (1.0 - extended_mask) * -10000.0
                else:
                    extended_mask = attn_mask[:, None, None, :].to(dtype=next(self.parameters()).dtype)
                    extended_mask = (1.0 - extended_mask) * -10000.0

                try:
                    encoder_outputs = bert_module.encoder(embedding_output, attention_mask=extended_mask, return_dict=True)
                except TypeError:
                    encoder_outputs = bert_module.encoder(embedding_output, attention_mask=attn_mask, return_dict=True)

                sequence_output = encoder_outputs.last_hidden_state if hasattr(encoder_outputs, 'last_hidden_state') else encoder_outputs[0]

            base_text_output = SimpleNamespace(last_hidden_state=sequence_output)

        # 6) slice prompt hidden states from Qformer outputs and project to LM
        text_len = base_input_embeddings.size(1)
        prompt_start = text_len
        prompt_end = text_len + self.num_prompt_tokens
        prompt_qformer_hidden = base_text_output.last_hidden_state[:, prompt_start:prompt_end, :]  # (bs, num_prompt_tokens, qformer_hidden)

        proj_param = next(self.proj.parameters())
        proj_dev = proj_param.device
        proj_dtype = proj_param.dtype

        prompt_for_proj = prompt_qformer_hidden.to(device=proj_dev, dtype=proj_dtype)
        prompt_proj_lm = self.proj(prompt_for_proj)  # (bs, num_prompt_tokens, lm_n_embd) in proj_dtype

        # 7) get text LM embeddings for input tokens (to match device/dtype)
        inputs_txt_embeds = self.lm_model.transformer.wte(input_tokens.input_ids).to(device)
        lm_emb_dtype = inputs_txt_embeds.dtype

        if prompt_proj_lm.device != inputs_txt_embeds.device or prompt_proj_lm.dtype != lm_emb_dtype:
            prompt_proj_lm = prompt_proj_lm.to(device=inputs_txt_embeds.device, dtype=lm_emb_dtype)

        # 8) --- NEW: compute adapter-based LM tokens from visual intermediate layers (if available) ---
        if self.use_adapter and (vis_layer_outputs is not None) and hasattr(self, 'adapter_proj_to_lm'):
            adapter_tokens = []
            # iterate in same order as align_layer_idxs and vis_layer_outputs
            for idx, layer_out in zip(self.align_layer_idxs, vis_layer_outputs):
                # layer_out shape: bs, seq_len, vis_feat_dim
                layer_cls = layer_out[:, 0, :]  # take CLS token
                # project to embed_dim
                proj = self.vis_layer_projs[str(idx)](layer_cls)  # bs, embed_dim

                # adapter + residual if exists
                if str(idx) in self.vis_layer_adapters:
                    adapter_module = self.vis_layer_adapters[str(idx)]
                    adapter_out = adapter_module(proj)
                    proj = proj + adapter_out

                # normalize then project to LM embedding dim
                proj = F.normalize(proj, dim=-1)
                adapter_lm = self.adapter_proj_to_lm(proj)  # (bs, lm_n_embd)
                # ensure dtype/device matches LM embeddings
                if adapter_lm.device != inputs_txt_embeds.device or adapter_lm.dtype != lm_emb_dtype:
                    adapter_lm = adapter_lm.to(device=inputs_txt_embeds.device, dtype=lm_emb_dtype)
                adapter_tokens.append(adapter_lm.unsqueeze(1))  # (bs,1,lm_n_embd)

            # concat along sequence dim => (bs, n_layers, lm_n_embd)
            adapter_proj_lm = torch.cat(adapter_tokens, dim=1)
            adapter_atts = torch.ones(adapter_proj_lm.size()[:-1], dtype=torch.long).to(inputs_txt_embeds.device)
        else:
            # no adapter tokens
            adapter_proj_lm = torch.zeros((inputs_txt_embeds.size(0), 0, inputs_txt_embeds.size(-1)),
                                         device=inputs_txt_embeds.device, dtype=inputs_txt_embeds.dtype)
            adapter_atts = torch.zeros((inputs_txt_embeds.size(0), 0), dtype=torch.long).to(inputs_txt_embeds.device)

        # 9) Ensure inputs_img is in correct device/dtype
        inputs_img_lm = inputs_img.to(device=inputs_txt_embeds.device, dtype=lm_emb_dtype)

        # 10) Build final inputs_embeds for LM: [text embeddings] + [prompt_proj_lm] + [adapter_proj_lm] + [image embeddings]
        inputs_embeds = torch.cat([inputs_txt_embeds, prompt_proj_lm, adapter_proj_lm, inputs_img_lm], dim=1)

        # 11) Build attention mask accordingly: text_mask + prompt_mask + adapter_mask + img_mask
        prompt_mask_lm = torch.ones((bs_local, self.num_prompt_tokens), dtype=input_tokens.attention_mask.dtype).to(inputs_txt_embeds.device)
        attention_mask = torch.cat([input_tokens.attention_mask.to(inputs_txt_embeds.device),
                                    prompt_mask_lm,
                                    adapter_atts,
                                    atts_img.to(inputs_txt_embeds.device)], dim=1)

        # 12) Filler input ids to start generation (on correct device)
        filler_input_ids = torch.full((inputs_embeds.shape[0], 1), fill_value=self.lm_model.config.bos_token_id,
                                      dtype=torch.long, device=inputs_txt_embeds.device)

        # 13) Generate with LM (keep autocast)
        with self.maybe_autocast():
            outputs = self.lm_model.generate(
                filler_input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )

        # 14) decode outputs
        output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]
        return output_text