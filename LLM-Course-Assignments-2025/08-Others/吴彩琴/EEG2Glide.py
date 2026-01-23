import os,sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # 指向 llm4eeg-master
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)          # 插到最前面，保证优先搜到
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from eeg_models.EEGNet import Model
from eeg_models.DATASET import EEGDataset,Splitter
import torch.nn.functional as F
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--subject',type=int,default=1)
parser.add_argument('--pre_epoch',type=int,default=1)
# parser.add_argument()
opt=parser.parse_args()
pretrained_file='saved_models/EEG_sem_clip/subject_'+str(opt.subject)+'/'+'/EEGNet/epoch_'+str(opt.pre_epoch)+'.pth'
saved_path='saved_models/mlpmodels/subject_'+str(opt.subject)+'/'
if(os.path.isdir(saved_path)==0):
        os.makedirs(saved_path)
splits_path='Data/EEGDataset/block_splits_by_image_subject_'+str(opt.subject)+'.pth'

target_embeddings=torch.load('Data/EmbeddingFiles/glide_photolabel_embeddings.pth')
# (40,512*6)
f=open(saved_path+'results.txt','a')
time_low=40
time_high=480
batch_size=340
epochs=250
learning_rate=0.001
device='cuda' if torch.cuda.is_available() else 'cpu'


dataset = EEGDataset("EEGDataset/eeg_55_95_std.pth")
loaders = {split: DataLoader(Splitter(dataset, split_path=splits_path, split_num=0, split_name=split),
                             batch_size=batch_size, drop_last=False, shuffle=True) for split in
           ["train", "val", "test"]}



model = Model().cuda()
print(model)
model.load_state_dict(torch.load(pretrained_file))

mlp_model=nn.Sequential(
    nn.Linear(160,500),
    nn.ReLU(),
    nn.Linear(500,6*512)
)
mlp_model=mlp_model.cuda()
for i, (layer, param) in enumerate(model.named_parameters()): 
        param.requires_grad = False

optimizer=torch.optim.AdamW(mlp_model.parameters(), lr=learning_rate,weight_decay=0.0001)


losses_per_epoch = {"train": [], "val": [], "test": []}
accuracies_per_epoch = {"train": [], "val": [], "test": []}

best_accuracy = 0
best_accuracy_val = 0
best_epoch = 0
predicted_labels = []
correct_labels = []
for epoch in range(1, epochs):
    losses = {"train": 0, "val": 0, "test": 0}
    counts = {"train": 0, "val": 0, "test": 0}
    for split in ("train", "val", "test"):
        if split == "train":
            mlp_model.train()
            torch.set_grad_enabled(True)
        else:
            mlp_model.eval()
            torch.set_grad_enabled(False)
        torch.set_grad_enabled(True)
        for i, (input_eeg, target_label, idx, subject, img) in enumerate(loaders[split]):
            for j in range(7):
                input=input_eeg[:,:,:,0+40*j:160+40*j]
                target=torch.stack(list(target_embeddings[idx] for idx in target_label),dim=0)
                input = input.to("cuda")
                target = target.to("cuda")
                with torch.no_grad():
                    eeg_feature, output_label = model(input)
                out_emb=mlp_model(eeg_feature)

                
                loss=F.mse_loss(out_emb,target)
                losses[split] += loss.item()
                if split == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                counts[split] += 1
                
                
    TrL1, VL1, TeL1 = losses["train"] / counts["train"], losses["val"] / counts["val"], losses["test"] / counts["test"] 
    
    print(
        "Epoch {0}: TrL1={1:.4f}, VL1={2:.4f}, TeL1={3:.4f}".format(
            epoch,
            TrL1, VL1, TeL1))
    f.write("Epoch {0}: TrL1={1:.4f}, VL1={2:.4f}, TeL1={3:.4f}".format(
            epoch,
            TrL1, VL1, TeL1)+'\n')
    
    torch.save(mlp_model.state_dict(),saved_path+'/mlp_epoch_'+str(epoch)+'.pth')