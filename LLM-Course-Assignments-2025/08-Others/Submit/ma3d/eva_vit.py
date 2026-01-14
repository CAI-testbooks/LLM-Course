import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from timm.models.layers import drop_path, to_3tuple, trunc_normal_
from lavis.common.dist_utils import download_cached_file

def _c_c(u='', **k):
    return {'url': u, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None, 'crop_pct': .9, 'interpolation': 'bicubic', 'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5), **k}

class DP(nn.Module):
    def __init__(self, p=None):
        super(DP, self).__init__()
        self.p = p
    def forward(self, x):
        return drop_path(x, self.p, self.training)

class M(nn.Module):
    def __init__(self, i_f, h_f=None, o_f=None, a_l=nn.GELU, d=0.):
        super().__init__()
        o_f = o_f or i_f
        h_f = h_f or i_f
        self.f1 = nn.Linear(i_f, h_f)
        self.a = a_l()
        self.f2 = nn.Linear(h_f, o_f)
        self.d = nn.Dropout(d)
    def forward(self, x):
        x = self.f1(x)
        x = self.a(x)
        x = self.f2(x)
        x = self.d(x)
        return x

class A(nn.Module):
    def __init__(self, d, n_h=8, q_b=False, q_s=None, a_d=0., p_d=0., w_s=None, a_h_d=None):
        super().__init__()
        self.n_h = n_h
        h_d = d // n_h
        if a_h_d is not None:
            h_d = a_h_d
        a_h_d = h_d * self.n_h
        self.s = q_s or h_d ** -0.5
        self.qkv = nn.Linear(d, a_h_d * 3, bias=False)
        if q_b:
            self.q_b = nn.Parameter(torch.zeros(a_h_d))
            self.v_b = nn.Parameter(torch.zeros(a_h_d))
        else:
            self.q_b = None
            self.v_b = None
        if w_s:
            self.w_s = w_s
            self.n_r_d = (2 * w_s[0] - 1) * (2 * w_s[1] - 1) + 3
            self.r_p_b_t = nn.Parameter(torch.zeros(self.n_r_d, n_h))
            c_h = torch.arange(w_s[0])
            c_w = torch.arange(w_s[1])
            c = torch.stack(torch.meshgrid([c_h, c_w]))
            c_f = torch.flatten(c, 1)
            r_c = c_f[:, :, None] - c_f[:, None, :]
            r_c = r_c.permute(1, 2, 0).contiguous()
            r_c[:, :, 0] += w_s[0] - 1
            r_c[:, :, 1] += w_s[1] - 1
            r_c[:, :, 0] *= 2 * w_s[1] - 1
            r_p_i = torch.zeros(size=(w_s[0] * w_s[1] + 1, ) * 2, dtype=r_c.dtype)
            r_p_i[1:, 1:] = r_c.sum(-1)
            r_p_i[0, 0:] = self.n_r_d - 3
            r_p_i[0:, 0] = self.n_r_d - 2
            r_p_i[0, 0] = self.n_r_d - 1
            self.register_buffer("r_p_i", r_p_i)
        else:
            self.w_s = None
            self.r_p_b_t = None
            self.r_p_i = None
        self.a_d = nn.Dropout(a_d)
        self.p = nn.Linear(a_h_d, d)
        self.p_d = nn.Dropout(p_d)
    def forward(self, x, r_p_b=None):
        B, N, C = x.shape
        q_b = None
        if self.q_b is not None:
            q_b = torch.cat((self.q_b, torch.zeros_like(self.v_b, requires_grad=False), self.v_b))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=q_b)
        qkv = qkv.reshape(B, N, 3, self.n_h, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.s
        at = (q @ k.transpose(-2, -1))
        if self.r_p_b_t is not None:
            r_p_b = self.r_p_b_t[self.r_p_i.view(-1)].view(self.w_s[0] * self.w_s[1] + 1, self.w_s[0] * self.w_s[1] + 1, -1)
            r_p_b = r_p_b.permute(2, 0, 1).contiguous()
            at = at + r_p_b.unsqueeze(0)
        if r_p_b is not None:
            at = at + r_p_b
        at = at.softmax(dim=-1)
        at = self.a_d(at)
        x = (at @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.p(x)
        x = self.p_d(x)
        return x

class B(nn.Module):
    def __init__(self, d, n_h, m_r=4., q_b=False, q_s=None, dr=0., a_d=0., d_p=0., i_v=None, a_l=nn.GELU, n_l=nn.LayerNorm, w_s=None, a_h_d=None):
        super().__init__()
        self.n1 = n_l(d)
        self.at = A(d, n_h=n_h, q_b=q_b, q_s=q_s, a_d=a_d, p_d=dr, w_s=w_s, a_h_d=a_h_d)
        self.d_p = DP(d_p) if d_p > 0. else nn.Identity()
        self.n2 = n_l(d)
        m_h_d = int(d * m_r)
        self.m = M(i_f=d, h_f=m_h_d, a_l=a_l, d=dr)
        if i_v is not None and i_v > 0:
            self.g1 = nn.Parameter(i_v * torch.ones((d)), requires_grad=True)
            self.g2 = nn.Parameter(i_v * torch.ones((d)), requires_grad=True)
        else:
            self.g1, self.g2 = None, None
    def forward(self, x, r_p_b=None):
        if self.g1 is None:
            x = x + self.d_p(self.at(self.n1(x), r_p_b=r_p_b))
            x = x + self.d_p(self.m(self.n2(x)))
        else:
            x = x + self.d_p(self.g1 * self.at(self.n1(x), r_p_b=r_p_b))
            x = x + self.d_p(self.g2 * self.m(self.n2(x)))
        return x

class PE(nn.Module):
    def __init__(self, i_s=224, p_s=16, i_c=3, e_d=768):
        super().__init__()
        i_s = to_3tuple(i_s)
        p_s = to_3tuple(p_s)
        n_p = (i_s[2] // p_s[2]) * (i_s[1] // p_s[1]) * (i_s[0] // p_s[0])
        self.p_s = (i_s[0] // p_s[0], i_s[1] // p_s[1], i_s[2] // p_s[2])
        self.i_s = i_s
        self.p_sz = p_s
        self.n_p = n_p
        self.pr = nn.Conv3d(i_c, e_d, kernel_size=p_s, stride=p_s)
    def forward(self, x):
        B, C, D, H, W = x.shape
        x = self.pr(x).flatten(2).transpose(1, 2)
        return x

class RPB(nn.Module):
    def __init__(self, w_s, n_h):
        super().__init__()
        self.w_s = w_s
        self.n_r_d = (2 * w_s[0] - 1) * (2 * w_s[1] - 1) + 3
        self.r_p_b_t = nn.Parameter(torch.zeros(self.n_r_d, n_h))
        c_h = torch.arange(w_s[0])
        c_w = torch.arange(w_s[1])
        c = torch.stack(torch.meshgrid([c_h, c_w]))
        c_f = torch.flatten(c, 1)
        r_c = c_f[:, :, None] - c_f[:, None, :]
        r_c = r_c.permute(1, 2, 0).contiguous()
        r_c[:, :, 0] += w_s[0] - 1
        r_c[:, :, 1] += w_s[1] - 1
        r_c[:, :, 0] *= 2 * w_s[1] - 1
        r_p_i = torch.zeros(size=(w_s[0] * w_s[1] + 1,) * 2, dtype=r_c.dtype)
        r_p_i[1:, 1:] = r_c.sum(-1)
        r_p_i[0, 0:] = self.n_r_d - 3
        r_p_i[0:, 0] = self.n_r_d - 2
        r_p_i[0, 0] = self.n_r_d - 1
        self.register_buffer("r_p_i", r_p_i)
    def forward(self):
        r_p_b = self.r_p_b_t[self.r_p_i.view(-1)].view(self.w_s[0] * self.w_s[1] + 1, self.w_s[0] * self.w_s[1] + 1, -1)
        return r_p_b.permute(2, 0, 1).contiguous()

class VT(nn.Module):
    def __init__(self, i_s=224, p_s=16, i_c=1, n_cl=1000, e_d=768, d=12, n_h=12, m_r=4., q_b=False, q_s=None, dr=0., a_dr=0., d_pr=0., n_l=nn.LayerNorm, i_v=None, u_a_p=True, u_r_p=False, u_s_r=False, u_m_p=True, i_sc=0.001, u_cp=False):
        super().__init__()
        self.i_s = i_s
        self.n_cl = n_cl
        self.e_d = e_d
        self.p_e_3 = PE(i_s=i_s, p_s=p_s, i_c=i_c, e_d=e_d)
        n_p = self.p_e_3.n_p
        self.cl_t = nn.Parameter(torch.zeros(1, 1, e_d))
        if u_a_p:
            self.p_e_3d = nn.Parameter(torch.zeros(1, n_p + 1, e_d))
        else:
            self.p_e_3d = None
        self.p_d = nn.Dropout(p=dr)
        if u_s_r:
            self.r_p_b = RPB(window_size=self.p_e_3.p_s, num_heads=n_h)
        else:
            self.r_p_b = None
        self.u_cp = u_cp
        d_p_r = [x.item() for x in torch.linspace(0, d_pr, d)]
        self.bl = nn.ModuleList([B(d=e_d, n_h=n_h, m_r=m_r, q_b=q_b, q_s=q_s, dr=dr, a_d=a_dr, d_p=d_p_r[i], n_l=n_l, i_v=i_v, w_s=self.p_e_3.p_s if u_r_p else None) for i in range(d)])
        if self.p_e_3d is not None:
            trunc_normal_(self.p_e_3d, std=.02)
        trunc_normal_(self.cl_t, std=.02)
        self.apply(self._i_w)
        self._f_i_w()
    def _f_i_w(self):
        def _res(p, l_i):
            p.div_(math.sqrt(2.0 * l_i))
        for l_i, l in enumerate(self.bl):
            _res(l.at.p.weight.data, l_i + 1)
            _res(l.m.f2.weight.data, l_i + 1)
    def _i_w(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def f_f(self, x):
        x = self.p_e_3(x)
        b, s, _ = x.size()
        cl = self.cl_t.expand(b, -1, -1)
        x = torch.cat((cl, x), dim=1)
        if self.p_e_3d is not None:
            x = x + self.p_e_3d
        x = self.p_d(x)
        r_p = self.r_p_b() if self.r_p_b is not None else None
        for bl in self.bl:
            if self.u_cp:
                x = cp.checkpoint(bl, x, r_p)
            else:
                x = bl(x, r_p)
        return x
    def forward(self, x):
        return self.f_f(x)

def _c_w_16(m: nn.Module):
    def _c(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()
    m.apply(_c)

def c_e_v_g(i_s=256, p_s=28, d_pr=0.4, u_cp=False, pr="fp16"):
    m = VT(i_s=i_s, p_s=p_s, u_m_p=False, e_d=1408, d=39, n_h=1408//88, m_r=4.3637, q_b=True, d_pr=d_pr, n_l=partial(nn.LayerNorm, eps=1e-6), u_cp=u_cp)
    u = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth"
    c_f = download_cached_file(u, check_hash=False, progress=True)
    s_d = torch.load(c_f, map_location="cpu")
    m.load_state_dict(s_d, strict=False)
    if pr == "fp16":
        _c_w_16(m)
    return m