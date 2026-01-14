import math
from functools import partial
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as ac
from torch.nn import functional as F
from transformers import T5TokenizerFast
from align3d.blip2 import Blip2Base
from align3d.modeling_ma3d_t5 import MA3D_T5_Head, T5Config
from align3d.eva_vit import c_e_v_g


class LN(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        o_t = x.dtype
        r = super().forward(x.type(torch.float32))
        return r.type(o_t)


class MA3D_T5(Blip2Base):
    def __init__(
            self,
            v_m="eva_clip_g",
            i_s=224,
            p_s=32,
            d_p_r=0,
            u_g_c=False,
            v_pr="fp16",
            f_v=True,
            n_q_t=32,
            t5_m="google/flan-t5-xl",
            m_t_l=60,
            e_d=256,
    ):
        super().__init__()
        self.tok = self.init_tokenizer()
        self.v_enc, self.ln_v = self._i_v_e(v_m, i_s, p_s, d_p_r, u_g_c, v_pr)
        if f_v:
            for n, p in self.v_enc.named_parameters():
                if '3d' not in n: p.requires_grad = False

        self.aq_m, self.q_t = self.init_Qformer(n_q_t, self.v_enc.num_features)
        self.aq_m.cls = None
        self.aq_m.bert.embeddings.word_embeddings = None
        self.aq_m.bert.embeddings.position_embeddings = None

        self.t5_tok = T5TokenizerFast.from_pretrained(t5_m)
        t5_cfg = T5Config.from_pretrained(t5_m)
        t5_cfg.dense_act_fn = "gelu"
        t5_cfg.output_attentions = True
        self.t5_m = MA3D_T5_Head.from_pretrained(t5_m, config=t5_cfg)

        for n, p in self.t5_m.named_parameters():
            p.requires_grad = False
            p.data = p.data.bfloat16()

        self.v_p = nn.Linear(self.aq_m.config.hidden_size, e_d)
        self.t_p = nn.Linear(self.aq_m.config.hidden_size, e_d)
        self.q_p = nn.Linear(self.aq_m.config.hidden_size, e_d)
        self.t5_p = nn.Linear(self.aq_m.config.hidden_size, self.t5_m.config.hidden_size)
        self.tmp = nn.Parameter(0.07 * torch.ones([]))
        self.m_t_l = m_t_l

    def _i_v_e(self, m_n, i_s, p_s, d_r, u_c, pr):
        v_e = c_e_v_g(i_s, p_s, d_r, u_c, pr)
        ln_v = LN(v_e.num_features)
        return v_e, ln_v

    def forward(self, smp):
        im = smp["images"].cuda().half()
        tx, qn, aw, qa, tq = [], [], [], [], []
        b_s = len(smp['reports'])
        for b in range(b_s):
            d = smp['reports'][b]
            if 'The diagnosis is' in d:
                tx.append(d.split('The diagnosis is ')[0])
                qn.append('What will this subject be diagnosed with?')
                lbl = d.split('The diagnosis is ')[1].split('.')[0]
                lbl = lbl.replace('AD', 'Dementia').replace('NC', 'Not demented').replace('MCI',
                                                                                          'mild cognitive impairment (MCI)')
                aw.append(lbl)
                qa.append('Question: What will this subject be diagnosed with? Answer: ' + lbl)
                tq.append(
                    d.split('The diagnosis is ')[0] + 'Question: What will this subject be diagnosed with? Answer: ')
            else:
                tx.append(d);
                qn.append('What will this subject be diagnosed with?')
                aw.append('');
                qa.append('Question: What will this subject be diagnosed with? Answer: ')
                tq.append(
                    d.split('The diagnosis is ')[0] + 'Question: What will this subject be diagnosed with? Answer: ')

        with self.maybe_autocast():
            i_eb = self.ln_v(self.v_enc(im))
        i_at = torch.ones(i_eb.size()[:-1], dtype=torch.long).to(im.device)
        q_tk = self.q_t.expand(i_eb.shape[0], -1, -1)
        q_o = self.aq_m.bert(query_embeds=q_tk, encoder_hidden_states=i_eb, encoder_attention_mask=i_at,
                             return_dict=True)
        t_tk = self.tok(tx, padding="max_length", truncation=True, max_length=self.m_t_l, return_tensors="pt").to(
            im.device)
        t_o = self.aq_m.bert(t_tk.input_ids, attention_mask=t_tk.attention_mask, return_dict=True)
        qa_tk = self.tok(qa, padding="max_length", truncation=True, max_length=self.m_t_l, return_tensors="pt").to(
            im.device)
        qa_o = self.aq_m.bert(qa_tk.input_ids, attention_mask=qa_tk.attention_mask, return_dict=True)
        i_f = F.normalize(self.v_p(q_o.last_hidden_state), dim=-1)
        t_f = F.normalize(self.t_p(t_o.last_hidden_state[:, 0, :]), dim=-1)
        qa_f = F.normalize(self.q_p(qa_o.last_hidden_state[:, 0, :]), dim=-1)
        s_q2t = torch.matmul(i_f.unsqueeze(1), t_f.unsqueeze(-1)).squeeze()
        s_i2t, _ = s_q2t.max(-1)
        s_i2t = s_i2t / self.tmp
        s_t2q = torch.matmul(t_f.unsqueeze(1).unsqueeze(1), i_f.permute(0, 2, 1)).squeeze()
        s_t2i, _ = s_t2q.max(-1)
        s_t2i = s_t2i / self.tmp
        tgs = torch.linspace(0, b_s - 1, b_s, dtype=int).to(im.device)
        l_itc = (F.cross_entropy(s_i2t, tgs, label_smoothing=0.1) + F.cross_entropy(s_t2i, tgs,
                                                                                    label_smoothing=0.1)) / 2
        s_q2qa = torch.matmul(i_f.unsqueeze(1), qa_f.unsqueeze(-1)).squeeze()
        s_i2qa, _ = s_q2qa.max(-1)
        s_i2qa = s_i2qa / self.tmp
        s_qa2q = torch.matmul(qa_f.unsqueeze(1).unsqueeze(1), i_f.permute(0, 2, 1)).squeeze()
        s_qa2i, _ = s_qa2q.max(-1)
        s_qa2i = s_qa2i / self.tmp
        l_itc += (F.cross_entropy(s_i2qa, tgs, label_smoothing=0.1) + F.cross_entropy(s_qa2i, tgs,
                                                                                      label_smoothing=0.1)) / 2
        in_t5 = self.t5_p(q_o.last_hidden_state)
        at_t5 = torch.ones(in_t5.size()[:-1], dtype=torch.long).to(im.device)
        with self.maybe_autocast(dtype=torch.bfloat16):
            i_tks = self.t5_tok(tq, padding="longest", truncation=True, max_length=self.m_t_l, return_tensors="pt").to(
                im.device)
            o_tks = self.t5_tok(aw, padding="longest", truncation=True, max_length=self.m_t_l, return_tensors="pt").to(
                im.device)
            e_ats = torch.cat([i_tks.attention_mask, at_t5], dim=1)
            tgs_lm = o_tks.input_ids.masked_fill(o_tks.input_ids == self.t5_tok.pad_token_id, -100)
            i_ebs = self.t5_m.encoder.embed_tokens(i_tks.input_ids)
            i_ebs = torch.cat([i_ebs, in_t5], dim=1)
            outs = self.t5_m(inputs_embeds=i_ebs, attention_mask=e_ats, decoder_attention_mask=o_tks.attention_mask,
                             return_dict=True, labels=tgs_lm)
            l_lm = outs.loss
            loss = l_itc + l_lm
        return {"loss": loss}

    @torch.no_grad()
    def generate(self, smp, u_n_s=False, n_b=5, m_l=60, m_n_l=1, t_p=0.9, r_p=1.0, l_p=1.0, n_c=1, tm=1, dev='cuda:0'):
        i_tks = self.t5_tok(smp["prompt"], padding="longest", return_tensors="pt").to(dev)
        if 'images' in smp.keys():
            im = smp["images"]
            with self.maybe_autocast():
                i_eb = self.ln_v(self.v_enc(im))
            i_eb = i_eb.float()
            i_at = torch.ones(i_eb.size()[:-1], dtype=torch.long).to(im.device)
            q_tk = self.q_t.expand(i_eb.shape[0], -1, -1)
            q_o = self.aq_m.bert(query_embeds=q_tk, encoder_hidden_states=i_eb, encoder_attention_mask=i_at,
                                 return_dict=True)
            i_t5 = self.t5_p(q_o.last_hidden_state)
            at_t5 = torch.ones(i_t5.size()[:-1], dtype=torch.long).to(im.device)
            e_ats = torch.cat([i_tks.attention_mask, at_t5], dim=1)
            with self.maybe_autocast(dtype=torch.bfloat16):
                eb_t5 = self.t5_m.encoder.embed_tokens(i_tks.input_ids)
                eb_t5 = torch.cat([eb_t5, i_t5], dim=1)
        else:
            e_ats = i_tks.attention_mask
            with self.maybe_autocast(dtype=torch.bfloat16):
                eb_t5 = self.t5_m.encoder.embed_tokens(i_tks.input_ids)
        outs = self.t5_m.generate(inputs_embeds=eb_t5, attention_mask=e_ats, do_sample=u_n_s, top_p=t_p, temperature=tm,
                                  num_beams=1, max_new_tokens=m_l, min_length=m_n_l, repetition_penalty=r_p,
                                  length_penalty=l_p, num_return_sequences=n_c, return_dict_in_generate=True)
        return self.t5_tok.batch_decode(outs.sequences, skip_special_tokens=True)