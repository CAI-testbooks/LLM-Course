import os,sys
sys.path.append(os.getcwd())
import torch;
torch.utils.backcompat.broadcast_warning.enabled = True
from torch.utils.data import DataLoader
from eeg_models.DATASET import EEGDataset, Splitter
from eeg_models.EEGNet import Model
import torch.nn.functional as F
import torch.optim
import argparse

parser = argparse.ArgumentParser(description="Template")
parser.add_argument('-ed', '--eeg-dataset', default="Data/EEGDataset/eeg_55_95_std.pth", help="EEG dataset path")
parser.add_argument('-sub', '--subject', type=int,default=1,help="choose a subject from 1 to 6, default is 0 (all subjects)")
parser.add_argument('--start_epoch',type=int,default=1)
opt = parser.parse_args()
start_epoch=opt.start_epoch
splits_path='Data/EEGDataset/block_splits_by_image_subject_'+str(opt.subject)+'.pth'
saved_path='saved_models/EEG_vis_clip/subject_'+str(opt.subject)+'/'
if (os.path.exists(saved_path) == 0):
    os.makedirs(saved_path)
f=open(saved_path+'/results.txt','a')

model = Model().cuda()

import numpy as np
def clip_loss(fea_1,fea_2):
    features_1 = fea_1 / fea_1.norm(dim=1, keepdim=True)
    features_2 = fea_2 / fea_2.norm(dim=1, keepdim=True)
    logit_scale=torch.ones([])*np.log(1 / 0.07)
    logit_scale = logit_scale.exp()
    logits_per_fea1 = logit_scale * features_1 @ features_2.t()
    logits_per_fea2 = logits_per_fea1.t()
    return logits_per_fea1,logits_per_fea2


time_low=40
time_high=480
batch_size=32
learning_rate=0.001
epochs=200
saveCheck=2
# Dataset class

dataset = EEGDataset("Data/EEGDataset/eeg_55_95_std.pth")
loaders = {split: DataLoader(Splitter(dataset, split_path=splits_path, split_num=0, split_name=split),
                             batch_size=batch_size, drop_last=True, shuffle=True) for split in
           ["train", "val", "test"]}


optimizer = getattr(torch.optim, "AdamW")(model.parameters(), lr=learning_rate)
vision_feature=torch.load('Data/EmbeddingFiles/pca_20').float()
vision_feature=vision_feature/torch.norm(vision_feature,dim=1,keepdim=True)

print(model)
losses_per_epoch = {"train": [], "val": [], "test": []}
accuracies_per_epoch = {"train": [], "val": [], "test": []}

best_accuracy = 0
best_accuracy_val = 0
best_epoch = 0
predicted_labels = []
correct_labels = []
for epoch in range(start_epoch, start_epoch+epochs):
    losses = {"train": 0, "val": 0, "test": 0}
    accuracies = {"train": 0, "val": 0, "test": 0}
    counts = {"train": 0, "val": 0, "test": 0}
    #for split in ("train", "val", "test"):
    for split in (["train","val","test"]):
        if split == "train":
            model.train()
            torch.set_grad_enabled(True)
        else:
            model.eval()
            torch.set_grad_enabled(False)
        
        
        for i, (input_eeg, target_label, idx, subject, img) in enumerate(loaders[split]):
            for j in range(7):
                input=input_eeg[:,:,:,0+40*j:160+40*j]
                input = input.to("cuda")
                target=torch.stack(list(vision_feature[c] for c in img),dim=0)
                target = target.to("cuda")
                output = model(input)
                logits_eeg, logits_emb = clip_loss(output, target)
                labels = torch.arange(target.shape[0], device='cuda').long()
                loss = ((F.cross_entropy(logits_emb, labels) +F.cross_entropy(logits_eeg, labels)) / 2)*0.8+0.2*F.mse_loss(output,target)
                losses[split] += loss.item()
                counts[split] += 1
                if split == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
    print(
        "Epoch {0}: TrL={1:.4f}, VL={2:.4f}, TeL={3:.4f}".format(
            epoch,
            losses["train"]/ counts["train"],
            losses["val"]/ counts["val"],
            losses["test"]/ counts["test"]))


    f.write("Epoch {0}: TrL={1:.4f}, VL={2:.4f}, TeL={3:.4f}".format(
            epoch,
            losses["train"]/ counts["train"],
            losses["val"]/ counts["val"],
            losses["test"]/ counts["test"])+'\n')
    if epoch % saveCheck == 0:
        torch.save(model.state_dict(), saved_path+'/epoch_%d.pth' % (epoch))
f.close()