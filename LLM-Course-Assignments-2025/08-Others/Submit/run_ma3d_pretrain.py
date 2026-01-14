import os, random, torch
import numpy as np
from torch.utils.data import DataLoader
from align3d.modeling_ma3d_t5 import MA3D_T5
from align3d.modeling_ma3d_biolm import MA3D_BioLM
from align3d.dataset import MM_CAD_Dataset, ZSEvalSet
from align3d.dataset import MM_Collator, ZSEvalCollator
from align3d.engine import Trainer

def set_env(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed(s)
    os.environ.update({'PYTHONHASHSEED': str(s), 'TOKENIZERS_PARALLELISM': 'false', 'CUDA_VISIBLE_DEVICES': '0'})

set_env()
dev = "cuda" if torch.cuda.is_available() else "cpu"

cfg = {'eps': 100, 'wu': 0.1, 'lr': 2e-5, 'wd': 1e-4, 'bs': 7}

tr_d = ['ADNI-train', 'NACC-train', 'OASIS1-aligned_norm-train', 'OASIS1-aligned_orig-train',
        'OASIS1-norm-train', 'OASIS1-orig-train', 'OASIS2-train']
te_d = ['NACC-test']

tr_ld = DataLoader(MM_CAD_Dataset(datalist=tr_d), batch_size=cfg['bs'],
                   collate_fn=MM_Collator(), shuffle=True, pin_memory=True, num_workers=4, drop_last=True)
te_ld = DataLoader(ZSEvalSet(datalist=te_d), batch_size=cfg['bs'],
                   collate_fn=ZSEvalCollator(), shuffle=False, pin_memory=True, num_workers=4)

run_t5, run_biolm = False, True

if run_t5:
    m = MA3D_T5(t5_model="google/flan-t5-xl").to(dev)
    out = './ckpts/ma3d_pretrain/t5'
    Trainer().train(m, tr_ld, te_ld, warmup_ratio=cfg['wu'], epochs=cfg['eps'],
                    optimizer_params={'lr': cfg['lr']}, output_path=out, weight_decay=cfg['wd'], use_amp=True)

if run_biolm:
    m = MA3D_BioLM(lm_model="stanford-crfm/BioMedLM").to(dev)
    out = './ckpts/ma3d_pretrain/biolm'
    Trainer().train(m, tr_ld, te_ld, warmup_ratio=cfg['wu'], epochs=cfg['eps'],
                    optimizer_params={'lr': cfg['lr']}, output_path=out, weight_decay=cfg['wd'], use_amp=True)