import logging
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as ac
from torch.nn import functional as F
from align3d.blip2 import Blip2Base
from align3d.modeling_ma3d_biolm import MA3D_BioLM_Head
from align3d.eva_vit import c_e_v_g
from transformers import GPT2Tokenizer

class LN(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        o_t = x.dtype
        r = super().forward(x.type(torch.float32))
        return r.type(o_t)

class MA3D_BioLM(Blip2Base):
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
        l_m="stanford-crfm/BioMedLM",
        p="",
        m_t_l=100,
        a_l=False,
        e_d=256,
    ):
        super().__init__()
        self.v_enc, self.ln_v = self._i_v_e(v_m, i_s, p_s, d_p_r, u_g_c, v_pr)
        if f_v:
            for n, p in self.v_enc.named_parameters():
                if '3d' not in n:
                    p.requires_grad = False
        self.aq_m, self.q_t = self.init_Qformer(n_q_t, self.v_enc.num_features)
        self.aq_m.cls = None
        self.aq_m.bert.embeddings.word_embeddings = None
        self.aq_m.bert.embeddings.position_embeddings = None
        self.tok = GPT2Tokenizer.from_pretrained(l_m, pad_token='<PAD>')
        self.lm = MA3D_BioLM_Head.from_pretrained(l_m, torch_dtype=torch.float16)
        for n, p in self.lm.named_parameters():
            p.requires_grad = False
        self.v_p = nn.Linear(self.aq_m.config.hidden_size, e_d)
        self.t_p = nn.Linear(self.aq_m.config.hidden_size, e_d)
        self.q_p = nn.Linear(self.aq_m.config.hidden_size, e_d)
        self.pr = nn.Linear(self.aq_m.config.hidden_size, self.lm.config.n_embd)
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
                lbl = lbl.replace('AD','Dementia').replace('Demented','Dementia').replace('NC','Not demented').replace('CN','Not demented').replace('Nondemented','Not demented').replace('control','Not demented').replace('MCI','mild cognitive impairment (MCI)')
                aw.append(lbl)
                qa.append('Question: What will this subject be diagnosed with? Answer: ' + lbl)
                tq.append(d.split('The diagnosis is ')[0] + 'Question: What will this subject be diagnosed with? Answer: ')
            else:
                tx.append(d)
                qn.append('What will this subject be diagnosed with?')
                aw.append('')
                qa.append('Question: What will this subject be diagnosed with? Answer: ')
                tq.append(d.split('The diagnosis is ')[0] + 'Question: What will this subject be diagnosed with? Answer: ')
        with self.maybe_autocast():
            i_eb = self.ln_v(self.v_enc(im))
        i_at = torch.ones(i_eb.size()[:-1], dtype=torch.long).to(im.device)
        q_tk = self.q_t.expand(i_eb.shape[0], -1, -1)
        q_o = self.aq_m.bert(query_embeds=q_tk, encoder_hidden_states=i_eb, encoder_attention_mask=i_at, return_dict=True)
        t_tk = self.tok(tx, padding="max_length", truncation=True, max_length=self.m_t_l, return_tensors="pt").to(im.device)
        t_o = self.aq_m.bert(t_tk.input_ids, attention_mask=t_tk.attention_mask, return_dict=True)
        qa_tk = self.tok(qa, padding="max_length", truncation=True, max_length=self.m_t_l, return_tensors="pt").to(im.device)
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
        l_itc = (F.cross_entropy(s_i2t, tgs, label_smoothing=0.1) + F.cross_entropy(s_t2i, tgs, label_smoothing=0.1)) / 2
        s_q2qa = torch.matmul(i_f.unsqueeze(1), qa_f.unsqueeze(-1)).squeeze()
        s_i2qa, _ = s_q2qa.max(-1)
        s_i2qa = s_i2qa / self.tmp
        s_qa2q = torch.matmul(qa_f.unsqueeze(1).unsqueeze(1), i_f.permute(0, 2, 1)).squeeze()
        s_qa2i, _ = s_qa2q.max(-1)
        s_qa2i = s_qa2i / self.tmp
        l_itc += (F.cross_entropy(s_i2qa, tgs, label_smoothing=0.1) + F.cross_entropy(s_qa2i, tgs, label_smoothing=0.1)) / 2
        im_eb = self.pr(q_o.last_hidden_state)
        at_im = torch.ones(im_eb.size()[:-1], dtype=torch.long).to(im.device)
        self.tok.padding_side = "right"
        in_tk = self.tok(tq, padding="longest", truncation=True, max_length=self.m_t_l, return_tensors="pt").to(im.device)
        ou_tk = self.tok(aw, padding="longest", truncation=True, max_length=self.m_t_l, return_tensors="pt").to(im.device)
        in_tg = in_tk.input_ids.masked_fill(in_tk.input_ids == self.tok.pad_token_id, -100)
        ou_tg = ou_tk.input_ids.masked_fill(ou_tk.input_ids == self.tok.pad_token_id, -100)
        em_i_tg = torch.ones(at_im.size(), dtype=torch.long).to(im.device).fill_(-100)
        tgs_all = torch.cat([in_tg, em_i_tg, ou_tg], dim=1)
        in_t_eb = self.lm.transformer.wte(in_tk.input_ids)
        ou_t_eb = self.lm.transformer.wte(ou_tk.input_ids)
        in_eb_all = torch.cat([in_t_eb, im_eb, ou_t_eb], dim=1)
        at_all = torch.cat([in_tk.attention_mask, at_im, ou_tk.attention_mask], dim=1)
        with self.maybe_autocast():
            outs = self.lm(inputs_embeds=in_eb_all, attention_mask=at_all, return_dict=True, labels=tgs_all)
        l_lm = outs.loss
        loss = l_itc + l_lm
        return {"loss": loss}

    @torch.no_grad()
    def generate(self, smp, dev='cuda:0'):
        i_tk = self.tok(smp["prompt"], padding="longest", truncation=True, return_tensors="pt", max_length=self.m_t_l).to(dev)
        im = smp["images"]
        with self.maybe_autocast():
            i_eb = self.ln_v(self.v_enc(im))
        i_eb = i_eb.float()
        i_at = torch.ones(i_eb.size()[:-1], dtype=torch.long).to(im.device)
        q_tk = self.q_t.expand(i_eb.shape[0], -1, -1)
        q_o = self.aq_m.bert(query_embeds=q_tk, encoder_hidden_states=i_eb, encoder_attention_mask=i_at, return_dict=True)
        i_im = self.pr(q_o.last_hidden_state)
        at_im = torch.ones(i_im.size()[:-1], dtype=torch.long).to(im.device)
        i_t_eb = self.lm.transformer.wte(i_tk.input_ids)
        eb_all = torch.cat([i_t_eb, i_im], dim=1)
        at_all = torch.cat([i_tk.attention_mask, at_im], dim=1)
        f_ids = torch.ones([eb_all.shape[0], 1], dtype=torch.long).to(im.device).fill_(self.lm.config.bos_token_id)
        with self.maybe_autocast():
            outs = self.lm.generate(f_ids, inputs_embeds=eb_all, attention_mask=at_all)
        ou_txt = self.tok.batch_decode(outs, skip_special_tokens=True)
        return [t.strip() for t in ou_txt]