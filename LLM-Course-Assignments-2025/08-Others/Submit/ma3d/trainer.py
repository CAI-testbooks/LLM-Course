import os, math, torch, transformers
from torch.optim import Optimizer
from typing import List, Dict, Type

class MA3D_Engine:
    def __init__(self, a=None):
        pass

    def exec_tr(self,
        m,
        dl,
        ev_dl,
        eps: int = 1,
        sh_type: str = 'WarmupCosine',
        wu_s: int = 10000,
        wu_r: float = 0.01,
        ckpt_dir: str = './ckpts/ma3d_pretrain',
        opt_cls: Type[Optimizer] = torch.optim.AdamW,
        opt_p: Dict[str, object]= {'lr': 2e-5},
        wd: float = 0.01,
        mgn: float = 1,
        amp: bool = False,
        acc_s: int = 1,
        ):
        self.acc_s = acc_s
        if amp:
            from torch.cuda.amp import autocast
            sc = torch.cuda.amp.GradScaler()

        st_p_e = len(dl)
        t_st = int(st_p_e * eps)
        wu_s = math.ceil(t_st * wu_r)

        p_opt = list(m.named_parameters())
        n_d = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        opt_g = [
            {'params': [p for n, p in p_opt if not any(nd in n for nd in n_d)], 'weight_decay': wd},
            {'params': [p for n, p in p_opt if any(nd in n for nd in n_d)], 'weight_decay': 0.0}
        ]

        opt = opt_cls(opt_g, **opt_p)
        sch = self._get_sch(opt, s_t=sh_type, wu=wu_s, tt=t_st)

        m = m.cuda()
        sk_s = False

        for ep in range(eps):
            it = iter(dl)
            for tr_it in range(st_p_e):
                m.zero_grad()
                m.train()
                d = next(it)

                if amp:
                    with autocast():
                        l = m(d)
                    l_v = l['loss']
                    s_b = sc.get_scale()
                    sc.scale(l_v).backward()
                    sc.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(m.parameters(), mgn)
                    sc.step(opt)
                    sc.update()
                    sk_s = sc.get_scale() != s_b
                else:
                    l = m(d)
                    l_v = l['loss'] / self.acc_s
                    l_v.backward()
                    torch.nn.utils.clip_grad_norm_(m.parameters(), mgn)
                    opt.step()

                print(f'EP[{ep}/{eps}]/IT[{tr_it}/{st_p_e}]: L: {l_v:.4f}')
                opt.zero_grad()

                if not sk_s:
                    sch.step()

                if (tr_it == (st_p_e - 1) and 't5' in ckpt_dir) or (tr_it == 1 and 'biolm' in ckpt_dir):
                    ev_it = iter(ev_dl)
                    n_ev = len(ev_dl)
                    for e_it in range(n_ev):
                        ev_d = next(ev_it)
                        ims = ev_d['images'].cuda().half()
                        aw, tq = [], []
                        b_sz = len(ev_d['reports'])
                        for b in range(b_sz):
                            doc = ev_d['reports'][b]
                            pre = doc.split('The diagnosis is ')[0] if 'The diagnosis is' in doc else doc
                            tq.append(pre + 'Question: What will this subject be diagnosed with? Answer: ')
                            if 'The diagnosis is' in doc:
                                lbl = doc.split('The diagnosis is ')[1].split('.')[0]
                                for k, v in {'AD':'Dementia','Demented':'Dementia','NC':'Not demented','CN':'Not demented','Nondemented':'Not demented','control':'Not demented','MCI':'mild cognitive impairment (MCI)'}.items():
                                    lbl = lbl.replace(k, v)
                                aw.append(lbl)
                            else:
                                aw.append('')
                        m.eval()
                        res = m.generate({"images": ims, 'prompt': tq})
                        for i in range(b_sz):
                            print(f'EV[{e_it}/{n_ev}][{i}/{b_sz}] RPT: {ev_d["reports"][i][:50]}...')
                            print(f'EV GT: {aw[i]} | PRD: {res[i]}')
                            print('-'*30)

            self._sv(m, ep, ckpt_dir)

    @staticmethod
    def _get_sch(opt, s_t: str, wu: int, tt: int):
        s_t = s_t.lower()
        if s_t == 'constantlr': return transformers.get_constant_schedule(opt)
        elif s_t == 'warmupconstant': return transformers.get_constant_schedule_with_warmup(opt, num_warmup_steps=wu)
        elif s_t == 'warmuplinear': return transformers.get_linear_schedule_with_warmup(opt, num_warmup_steps=wu, num_training_steps=tt)
        elif s_t == 'warmupcosine': return transformers.get_cosine_schedule_with_warmup(opt, num_warmup_steps=wu, num_training_steps=tt)
        elif s_t == 'warmupcosinewithhardrestarts': return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(opt, num_warmup_steps=wu, num_training_steps=tt)
        else: raise ValueError(f"ERR: {s_t}")

    def _sv(self, m, ep, dr):
        if not os.path.exists(dr): os.makedirs(dr)
        torch.save(m.state_dict(), os.path.join(dr, f'ep{ep}.pth'))