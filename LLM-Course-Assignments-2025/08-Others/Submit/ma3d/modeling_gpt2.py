import math
import os
import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import Conv1D
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

class MA3D_A(nn.Module):
    def __init__(self, cfg, is_cross=False, idx=None):
        super().__init__()
        mx_p = cfg.max_position_embeddings
        self.register_buffer("bias", torch.tril(torch.ones((mx_p, mx_p), dtype=torch.bool)).view(1, 1, mx_p, mx_p))
        self.register_buffer("m_b", torch.tensor(-1e4))
        self.e_d = cfg.hidden_size
        self.n_h = cfg.num_attention_heads
        self.h_d = self.e_d // self.n_h
        self.s_z = self.e_d
        self.s_a_w = cfg.scale_attn_weights
        self.is_c = is_cross
        self.s_a_i = cfg.scale_attn_by_inverse_layer_idx
        self.idx = idx
        self.r_u_a = cfg.reorder_and_upcast_attn
        if self.is_c:
            self.c_attn = Conv1D(2 * self.e_d, self.e_d)
            self.q_attn = Conv1D(self.e_d, self.e_d)
        else:
            self.c_attn = Conv1D(3 * self.e_d, self.e_d)
        self.c_p = Conv1D(self.e_d, self.e_d)
        self.a_dr = nn.Dropout(cfg.attn_pdrop)
        self.r_dr = nn.Dropout(cfg.resid_pdrop)

    def _at(self, q, k, v, m=None, hm=None):
        w = torch.matmul(q, k.transpose(-1, -2))
        if self.s_a_w:
            w = w / torch.full([], v.size(-1) ** 0.5, dtype=w.dtype, device=w.device)
        if self.s_a_i:
            w = w / float(self.idx + 1)
        if not self.is_c:
            ql, kl = q.size(-2), k.size(-2)
            c_m = self.bias[:, :, kl - ql : kl, :kl]
            mv = torch.full([], torch.finfo(w.dtype).min, dtype=w.dtype).to(w.device)
            w = torch.where(c_m, w.to(w.dtype), mv)
        if m is not None:
            w = w + m
        w = nn.functional.softmax(w, dim=-1).type(v.dtype)
        w = self.a_dr(w)
        if hm is not None:
            w = w * hm
        return torch.matmul(w, v), w

    def forward(self, x, p=None, m=None, hm=None, exs=None, em=None, uc=False, oa=False):
        if exs is not None:
            q = self.q_attn(x)
            k, v = self.c_attn(exs).split(self.s_z, dim=2)
            m = em
        else:
            q, k, v = self.c_attn(x).split(self.s_z, dim=2)
        q = q.view(q.size()[:-1] + (self.n_h, self.h_d)).permute(0, 2, 1, 3)
        k = k.view(k.size()[:-1] + (self.n_h, self.h_d)).permute(0, 2, 1, 3)
        v = v.view(v.size()[:-1] + (self.n_h, self.h_d)).permute(0, 2, 1, 3)
        if p is not None:
            pk, pv = p
            k, v = torch.cat((pk, k), dim=-2), torch.cat((pv, v), dim=-2)
        pr = (k, v) if uc else None
        o, w = self._at(q, k, v, m, hm)
        o = o.permute(0, 2, 1, 3).contiguous()
        o = self.c_p(o.view(o.size()[:-2] + (self.n_h * self.h_d,)))
        o = self.r_dr(o)
        res = (o, pr)
        if oa: res += (w,)
        return res

class MA3D_M(nn.Module):
    def __init__(self, i_s, cfg):
        super().__init__()
        e_d = cfg.hidden_size
        self.c_fc = Conv1D(i_s, e_d)
        self.c_p = Conv1D(e_d, i_s)
        self.act = ACT2FN[cfg.activation_function]
        self.dr = nn.Dropout(cfg.resid_pdrop)
    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_p(x)
        return self.dr(x)

class MA3D_B(nn.Module):
    def __init__(self, cfg, idx=None):
        super().__init__()
        h_s = cfg.hidden_size
        i_d = cfg.n_inner if cfg.n_inner is not None else 4 * h_s
        self.ln1 = nn.LayerNorm(h_s, eps=cfg.layer_norm_epsilon)
        self.at = MA3D_A(cfg, idx=idx)
        self.ln2 = nn.LayerNorm(h_s, eps=cfg.layer_norm_epsilon)
        if cfg.add_cross_attention:
            self.c_at = MA3D_A(cfg, is_cross=True, idx=idx)
            self.ln_c = nn.LayerNorm(h_s, eps=cfg.layer_norm_epsilon)
        self.m = MA3D_M(i_d, cfg)
    def forward(self, x, p=None, m=None, hm=None, exs=None, em=None, uc=False, oa=False):
        r = x
        x = self.ln1(x)
        at_o = self.at(x, p=p, m=m, hm=hm, uc=uc, oa=oa)
        x = at_o[0] + r
        outs = at_o[1:]
        if exs is not None:
            r = x
            x = self.ln_c(x)
            c_at_o = self.c_at(x, m=m, hm=hm, exs=exs, em=em, oa=oa)
            x = r + c_at_o[0]
            outs = outs + c_at_o[2:]
        r = x
        x = self.ln2(x)
        x = r + self.m(x)
        return (x, ) + (outs if uc else outs[1:])

class MA3D_MDL(PreTrainedModel):
    config_class = GPT2Config
    base_model_prefix = "transformer"
    def __init__(self, cfg):
        super().__init__(cfg)
        self.e_d = cfg.hidden_size
        self.wte = nn.Embedding(cfg.vocab_size, self.e_d)
        self.wpe = nn.Embedding(cfg.max_position_embeddings, self.e_d)
        self.dr = nn.Dropout(cfg.embd_pdrop)
        self.h = nn.ModuleList([MA3D_B(cfg, idx=i) for i in range(cfg.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.e_d, eps=cfg.layer_norm_epsilon)
        self.post_init()
    def forward(self, ids=None, pkv=None, m=None, tid=None, pid=None, hm=None, emb=None, exs=None, em=None, uc=None, oa=None, ohs=None, rd=None):
        oa = oa if oa is not None else self.config.output_attentions
        ohs = ohs if ohs is not None else self.config.output_hidden_states
        uc = uc if uc is not None else self.config.use_cache
        rd = rd if rd is not None else self.config.use_return_dict
        if ids is not None:
            sh = ids.size()
            ids = ids.view(-1, sh[-1])
            bs = ids.shape[0]
        elif emb is not None:
            sh = emb.size()[:-1]
            bs = emb.shape[0]
        dev = ids.device if ids is not None else emb.device
        if pid is None:
            pl = pkv[0][0].size(-2) if pkv is not None else 0
            pid = torch.arange(pl, sh[-1] + pl, dtype=torch.long, device=dev).unsqueeze(0).view(-1, sh[-1])
        if m is not None:
            m = m.view(bs, -1)[:, None, None, :]
            m = (1.0 - m.to(dtype=self.dtype)) * torch.finfo(self.dtype).min
        if self.config.add_cross_attention and exs is not None:
            ebs, esl, _ = exs.size()
            if em is None: em = torch.ones((ebs, esl), device=dev)
            em = self.invert_attention_mask(em)
        hm = self.get_head_mask(hm, self.config.n_layer)
        if emb is None: emb = self.wte(ids)
        x = self.dr(emb + self.wpe(pid))
        if tid is not None: x = x + self.wte(tid.view(-1, sh[-1]))
        prs, all_a, all_ca, all_h = ((), (), (), ())
        for i, (b, lp) in enumerate(zip(self.h, pkv or [None]*len(self.h))):
            if ohs: all_h += (x,)
            out = b(x, p=lp, m=m, hm=hm[i], exs=exs, em=em, uc=uc, oa=oa)
            x = out[0]
            if uc: prs += (out[1],)
            if oa:
                all_a += (out[2 if uc else 1],)
                if self.config.add_cross_attention: all_ca += (out[3 if uc else 2],)
        x = self.ln_f(x).view(sh + (x.size(-1),))
        if ohs: all_h += (x,)
        return (x, prs, all_h, all_a, all_ca)

class MA3D_LM(PreTrainedModel):
    config_class = GPT2Config
    def __init__(self, cfg):
        super().__init__(cfg)
        self.transformer = MA3D_MDL(cfg)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.post_init()
    def forward(self, ids=None, pkv=None, m=None, tid=None, pid=None, hm=None, emb=None, exs=None, em=None, lbl=None, uc=None, oa=None, ohs=None, rd=None):
        out = self.transformer(ids, pkv, m, tid, pid, hm, emb, exs, em, uc, oa, ohs, rd)
        x = out[0]
        lgt = self.lm_head(x)
        ls = None
        if lbl is not None:
            lbl = lbl.to(lgt.device)
            lgt = lgt[:, -lbl.shape[1]:, :]
            s_lgt = lgt[..., :-1, :].contiguous()
            s_lbl = lbl[..., 1:].contiguous()
            ls = torch.nn.CrossEntropyLoss()(s_lgt.view(-1, self.config.vocab_size), s_lbl.view(-1))
        return (ls, lgt) + out[1:]