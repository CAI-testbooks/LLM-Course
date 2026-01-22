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
parser.add_argument('-ed', '--eeg-dataset', default='Data/EEGDataset/eeg_55_95_std.pth', help="EEG dataset path")
parser.add_argument('-sub', '--subject', type=int,default=1,
                    help="choose a subject from 1 to 6, default is 0 (all subjects)")
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--pretrained_epoch',default=150)
parser.add_argument('--start_epoch',type=int,default=1)
opt = parser.parse_args()
start_epoch=opt.start_epoch
splits_path='Data/EEGDataset/block_splits_by_image_subject_'+str(opt.subject)+'.pth'
saved_path='saved_models/EEG_sem_classification/subject_'+str(opt.subject)+'/decay_'+str(opt.weight_decay)+'/'
if (os.path.exists(saved_path) == 0):
    os.makedirs(saved_path)

f=open(saved_path+'/results.txt','a')

time_low=40
time_high=480
batch_size=160
learning_rate=0.001
epochs=200
saveCheck=1


dataset = EEGDataset("Data/EEGDataset/eeg_55_95_std.pth")
loaders = {split: DataLoader(Splitter(dataset, split_path=splits_path, split_num=0, split_name=split),
                             batch_size=batch_size, drop_last=True, shuffle=True) for split in
           ["train", "val", "test"]}


model = Model().cuda()
model.load_state_dict(torch.load('saved_models/EEG_sem_clip/subject_'+str(opt.subject)+'/EEGNet/epoch_'+str(opt.pretrained_epoch)+'.pth'))

opt_param=[]
for i,(name,weight) in enumerate(model.named_parameters()):
    print(i,name)
    if(i>=16):
        opt_param.append(weight)

optimizer = torch.optim.AdamW(opt_param, lr=learning_rate, weight_decay=opt.weight_decay)

print(model)
losses_per_epoch = {"train": [], "val": [], "test": []}
accuracies_per_epoch = {"train": [], "val": [], "test": []}

best_accuracy = 0
best_accuracy_val = 0
best_epoch = 0
predicted_labels = []
correct_labels = []
for epoch in range(start_epoch, epochs + 1):
    losses = {"train": 0, "val": 0, "test": 0}
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
                target = target_label.to("cuda")
                target_label=target_label.to('cuda')
                output_emb, output_label = model(input)

                loss = F.cross_entropy(output_label,target_label)
                losses[split] += loss.item()
                _, pred = output_label.data.max(1)
                correct = pred.eq(target.data).sum().item()
                accuracy = correct / input.data.size(0)
                accuracies[split] += accuracy

                if split == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                counts[split] += 1


    TrL2, TrA2, VL2, VA2, TeL2, TeA2 = losses["train"] / counts["train"], accuracies["train"] / counts["train"], losses[
        "val"] / counts["val"], accuracies["val"] / counts["val"], losses["test"] / counts["test"], accuracies["test"] / \
                                 counts["test"]
    print(
        "Epoch {0}: TrL2={1:.4f}, TrA2={2:.4f}, VL2={3:.4f}, VA2={4:.4f}, TeL2={5:.4f}, TeA2={6:.4f}".format(
            epoch,
            TrL2, TrA2, VL2, VA2, TeL2, TeA2))



    f.write("Epoch {0}: TrL2={1:.4f}, TrA2={2:.4f}, VL2={3:.4f}, VA2={4:.4f}, TeL2={5:.4f}, TeA2={6:.4f}".format(
            epoch,
            TrL2, TrA2, VL2, VA2, TeL2, TeA2)+'\n')
    if epoch % saveCheck == 0:
        torch.save(model.state_dict(), saved_path+'epoch_%d.pth' % (epoch))

f.close()

