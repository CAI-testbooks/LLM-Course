import pdb, os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from model.modeling_medblip_biomedlm import MedBLIPModel_biomedlm
from data.dataset import BrainSCKValidDataset, BrainSCKValidCollator
from model.trainer import Trainer, adhd_eval
import argparse

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:28"
torch.cuda.empty_cache()

# set random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ['PYTHONASHSEED'] = str(seed)
os.environ['TOKENIZERS_PARALLELISM']='false'

lm_flan_t5 = False

def parse_args():
    parser = argparse.ArgumentParser(description='Test a mri brain cognition model based on MedBLIP.')
    parser.add_argument('--test_list', default='')
    parser.add_argument('--dataset_type', default='adhd')  ### adhd
    parser.add_argument('--gpu_id', default=0, type=int, help='gpu id for test')
    parser.add_argument('--load_model_path', default='', help='load model path') 
    
    args = parser.parse_args()
    
    return args

def main():

    print ('\n start model test ...')
    args = parse_args()  
    use_gpu_id = args.gpu_id
    test_datalist = args.test_list
    dataset_type = args.dataset_type
    load_model_path = args.load_model_path
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(use_gpu_id)
    torch.cuda.set_device(use_gpu_id)

    val_data = BrainSCKValidDataset(datalist=test_datalist)
    val_collate_fn = BrainSCKValidCollator()
    valloader = DataLoader(val_data,
        batch_size=4,
        collate_fn=val_collate_fn,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )
    print ('load valid data finish ...')

    print ('load biomed model ...')
    model = MedBLIPModel_biomedlm(
        lm_model="stanford-crfm/BioMedLM"
    )
    
    model.load_state_dict(torch.load(load_model_path, map_location='cpu'), strict=False)     
    print ('load model {} finish ...'.format(load_model_path))
        
    model.cuda()

    adhd_eval(model, valloader, dataset_type, gpu_id=use_gpu_id)

if __name__ == "__main__":
    main()  
    