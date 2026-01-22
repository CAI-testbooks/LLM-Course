import os,sys
sys.path.append(os.getcwd())
import torch;
torch.utils.backcompat.broadcast_warning.enabled = True
from torch.utils.data import DataLoader
from eeg_models.DATASET import EEGDataset, Splitter
from eeg_models.EEGNet import Model, EmbModel
import torch.nn.functional as F
import torch.optim
import argparse

parser = argparse.ArgumentParser(description="Template")
parser.add_argument('-ed', '--eeg-dataset', default="Data/EEGDataset/eeg_55_95_std.pth", help="EEG dataset path")
parser.add_argument('-sub', '--subject', type=int, default=1)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--pretrained_path')
parser.add_argument('--start_epoch',type=int,default=1)
opt = parser.parse_args()
start_epoch=opt.start_epoch
splits_path='Data/EEGDataset/block_splits_by_image_subject_'+str(opt.subject)+'.pth'
saved_path='saved_models/EEG_sem_clip/subject_'+str(opt.subject)+'/'
if (os.path.exists(saved_path) == 0):
    os.makedirs(saved_path+'EEGNet')
    os.makedirs(saved_path+'llamamlp')
f=open(saved_path+'/results.txt','a')

time_low=40
time_high=480
batch_size=16
learning_rate=0.001
epochs=150
saveCheck=1


import numpy as np
def clip_loss(fea_1,fea_2):
    features_1 = fea_1 / fea_1.norm(dim=1, keepdim=True)
    features_2 = fea_2 / fea_2.norm(dim=1, keepdim=True)
    logit_scale=torch.ones([])*np.log(1 / 0.07)
    logit_scale = logit_scale.exp()
    logits_per_fea1 = logit_scale * features_1 @ features_2.t()
    logits_per_fea2 = logits_per_fea1.t()
    return logits_per_fea1,logits_per_fea2

dataset = EEGDataset("EEGDataset/eeg_55_95_std.pth")
loaders = {split: DataLoader(Splitter(dataset, split_path=splits_path, split_num=0, split_name=split),
                             batch_size=batch_size, drop_last=True, shuffle=True) for split in
           ["train", "val", "test"]}

llama_emb=torch.load('Data/EmbeddingFiles/llama_desc_hiddenstate.pth')
model = Model().cuda()
llama_mlp=EmbModel().cuda()
opt_param1=[]
opt_param2=[]
for i,(name,weight) in enumerate(model.named_parameters()):
    # print(i,name)
    if(i<=15):
        opt_param1.append(weight)
    else:
        opt_param2.append(weight)
for name,weight in llama_mlp.named_parameters():
    opt_param1.append(weight)

optimizer1 = torch.optim.AdamW(opt_param1, lr=learning_rate, weight_decay=opt.weight_decay)
optimizer2 = torch.optim.AdamW(opt_param2, lr=learning_rate, weight_decay=0.01)

if(opt.pretrained_path is not None):
    model.load_state_dict(torch.load(opt.pretrained_path))

print(model)
losses_per_epoch = {"train": [], "val": [], "test": []}
accuracies_per_epoch = {"train": [], "val": [], "test": []}

best_accuracy = 0
best_accuracy_val = 0
best_epoch = 0
predicted_labels = []
correct_labels = []
for epoch in range(start_epoch, epochs + 1):
    losses1 = {"train": 0, "val": 0, "test": 0}
    losses2 = {"train": 0, "val": 0, "test": 0}
    counts = {"train": 0, "val": 0, "test": 0}
    accuracies = {"train": 0, "val": 0, "test": 0}
    for split in ("train", "val", "test"):
        # Set network mode
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
                target=torch.stack(list(llama_emb[idx] for idx in target_label),dim=0)
                target = target.to("cuda")
                target_label=target_label.to('cuda')
                output_emb, output_label = model(input)
                target=llama_mlp(target)
                labels = torch.arange(target.shape[0], device='cuda').long()

                logits_eeg, logits_emb = clip_loss(output_emb, target)

                loss1 = ((F.cross_entropy(logits_emb, labels) +F.cross_entropy(logits_eeg, labels)) / 2)*0.8+0.2*F.mse_loss(output_emb,target)

                losses1[split] += loss1.item()
                if split == "train":
                    optimizer1.zero_grad()
                    loss1.backward()
                    optimizer1.step()
                
                output_emb, output_label = model(input)
                loss2 = F.cross_entropy(output_label, target_label)
                losses2[split] += loss2.item()
                _, pred = output_label.data.max(1)
                correct = pred.eq(target_label.data).sum().item()
                accuracy = correct / input.data.size(0)
                accuracies[split] += accuracy
                if split == "train":
                    optimizer2.zero_grad()
                    loss2.backward()
                    optimizer2.step()
                counts[split] += 1


    TrL1, VL1, TeL1 = losses1["train"] / counts["train"], losses1["val"] / counts["val"], losses1["test"] / counts["test"] 
    TrL2, TrA2, VL2, VA2, TeL2, TeA2 = losses2["train"] / counts["train"], accuracies["train"] / counts["train"], losses2[
        "val"] / counts["val"], accuracies["val"] / counts["val"], losses2["test"] / counts["test"], accuracies["test"] / \
                                 counts["test"]
    print(
        "Epoch {0}: TrL1={1:.4f}, VL1={2:.4f}, TeL1={3:.4f}, TrL2={4:.4f}, TrA2={5:.4f}, VL2={6:.4f}, VA2={7:.4f}, TeL2={8:.4f}, TeA2={9:.4f}".format(
            epoch,
            TrL1, VL1, TeL1, TrL2, TrA2, VL2, VA2, TeL2, TeA2))



    f.write("Epoch {0}: TrL1={1:.4f}, VL1={2:.4f}, TeL1={3:.4f}, TrL2={4:.4f}, TrA2={5:.4f}, VL2={6:.4f}, VA2={7:.4f}, TeL2={8:.4f}, TeA2={9:.4f}".format(
            epoch,
            TrL1, VL1, TeL1, TrL2, TrA2, VL2, VA2, TeL2, TeA2)+'\n')
    if epoch % saveCheck == 0:
        torch.save(model.state_dict(), saved_path+'EEGNet/epoch_%d.pth' % (epoch))
        torch.save(llama_mlp.state_dict(), saved_path+'llamamlp/epoch_%d.pth' % (epoch))

f.close()

