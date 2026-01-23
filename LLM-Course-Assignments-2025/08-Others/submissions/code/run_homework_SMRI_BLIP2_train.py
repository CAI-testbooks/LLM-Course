import pdb, os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from model.modeling_medblip_biomedlm import MedBLIPModel_biomedlm
from data.dataset import BrainSCKTrainDataset, BrainSCKTrainCollator
from data.dataset import BrainSCKValidDataset, BrainSCKValidCollator
from model.trainer import Trainer
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

def parse_args():
    parser = argparse.ArgumentParser(description='Train a mri brain cognition model based on MedBLIP.')
    parser.add_argument('--dataset_type', default='adhd')  ### adhd
    parser.add_argument('--train_list', default='')
    parser.add_argument('--valid_list', default='')
    parser.add_argument('--gpu_id', default=0, type=int, help='gpu id for model train and valid')
    parser.add_argument('--max_epochs', default=100, type=int, help="max number of training epochs")
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--pretrain_weight', default='', help='pretrain weight path')
    parser.add_argument('--save_model_path', default='', help='model save path') 
    
    args = parser.parse_args()
    
    return args

def main():

    ### config setting ###
    args = parse_args()  
    dataset_type = args.dataset_type 
    use_gpu_id = args.gpu_id
    train_datalist = args.train_list
    val_datalist = args.valid_list
    batch_size = args.batch_size
    max_epochs = args.max_epochs
    model_save_path = args.save_model_path
    pretrain_weight = args.pretrain_weight
    print(args)

    torch.cuda.set_device(use_gpu_id)

    ######### step 1 #########
    print ('step 1. load train and valid dataset') 
    traindata = BrainSCKTrainDataset(datalist=train_datalist, dataset_type=dataset_type)
    train_collate_fn = BrainSCKTrainCollator()
    trainloader = DataLoader(traindata,
        batch_size=batch_size,
        collate_fn=train_collate_fn,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=True
    )

    print ('load train data finish ...')
    ### hcp pretrain stage do not need valid
    val_data = BrainSCKValidDataset(datalist=val_datalist)
    val_collate_fn = BrainSCKValidCollator()
    valloader = DataLoader(val_data,
        batch_size=batch_size,
        collate_fn=val_collate_fn,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )

    print ('load valid data finish ...')

    ######### step 2 #########
    print ('step 2. load large model')
    model = MedBLIPModel_biomedlm(
        lm_model="stanford-crfm/BioMedLM"
    )
    ### load pretrain weight from HCP ###
    if pretrain_weight != '' and os.path.exists(pretrain_weight):
        model.load_state_dict(torch.load(pretrain_weight, map_location='cpu'), strict=False)
        print ('load blip model {} finish ...'.format(pretrain_weight))
        
    model.cuda()

    ######### step 3 #########
    print ('step 3. start model training') 
    train_config = {
        'num_epochs': max_epochs,
        'warmup': 0.1,
        'lr': 2e-5,
        'weight_decay': 1e-4,
        'eval_batch_size': 8,
        'eval_steps': 1000,
        'save_steps': 1000,
    }

    trainer = Trainer()
    trainer.train(
        model,
        traindata,
        trainloader,
        valloader,
        use_gpu_id,
        dataset_type,
        warmup_ratio=train_config['warmup'],
        epochs=train_config['num_epochs'],
        optimizer_params={'lr':train_config['lr']},
        output_path=model_save_path,
        weight_decay=train_config['weight_decay'],
        use_amp=False,
        accumulation_steps=1,
    )

if __name__ == "__main__":
    main()    