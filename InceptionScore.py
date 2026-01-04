import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
from torchvision.models.inception import inception_v3
import numpy as np
from tqdm import tqdm
from PIL import Image
import os
from scipy.stats import entropy
import argparse
 
# python IS.py --input_image_dir ./input_images
 
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input_image_dir', type=str, default='generated_images')
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--device', type=str, choices=["cuda:0", "cpu"], default="cuda:0")
args = parser.parse_args()

mean_inception = [0.485, 0.456, 0.406]
std_inception = [0.229, 0.224, 0.225]
 
 
def imread(filename):
    """
    Loads an image file into a (height, width, 3) uint8 ndarray.
    """
    return np.asarray(Image.open(filename), dtype=np.uint8)[..., :3]
 
 
def inception_score(filepath,batch_size=args.batch_size, resize=True, splits=1):
    # Set up dtype
    device = torch.device(args.device)  # you can change the index of cuda
    # Load inception model
    # inception_model = inception_v3(transform_input=False).to(device)
    # inception_model.load_state_dict(torch.load('net_models/inception_v3_google-0cc3c7bd.pth'))
    inception_model = inception_v3(transform_input=False,pretrained=True).to(device)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).to(device)
 
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()
 
    # Get predictions using pre-trained inception_v3 model
    print('Computing predictions using inception v3 model')
 
    files = readDir(filepath)
    
    N = len(files)
    preds = np.zeros((N, 1000))
    if batch_size > N:
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
 
    for i in tqdm(range(0, N, batch_size)):
        start = i
        end = i + batch_size
        images = np.array([imread(str(f)).astype(np.float32)
                           for f in files[start:end]])
 
        # Reshape to (n_images, 3, height, width)
        images = images.transpose((0, 3, 1, 2))
        images /= 255
 
        batch = torch.from_numpy(images).type(torch.FloatTensor)
        batch = batch.to(device)
        preds[i:i + batch_size] = get_pred(batch)
 
    # assert batch_size > 0
    # assert N > batch_size
 
    # Now compute the mean KL Divergence
    print('Computing KL Divergence')
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]  # split the whole data into several parts
        py = np.mean(part, axis=0)  # marginal probability
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]  # conditional probability
            scores.append(entropy(pyx, py))  # compute divergence
        split_scores.append(np.exp(scores))
    return np.max(split_scores), np.mean(split_scores)
 
 
def readDir(dirPath):
    allFiles = []
    if os.path.isdir(dirPath):
        fileList = os.listdir(dirPath)
        for f in fileList:
            f = dirPath + '/' + f
            if os.path.isdir(f):
                subFiles = readDir(f)
                allFiles = subFiles + allFiles
            else:
                if(f.split('/')[-1][0]=='g'):
                    continue
                allFiles.append(f)
        return allFiles
    else:
        return 'Error,not a dir'
 
 
if __name__ == '__main__':
    f=open('metrics/ISintra_results.txt','a')
    f.write(args.input_image_dir+'\n')
    for subject in range(1,7):
        genpath=os.path.join(args.input_image_dir,str(subject))
        isresult=0
        for subdir in os.listdir(genpath):
            MAX, IS = inception_score(filepath=os.path.join(genpath,subdir),splits=1)
            isresult+=IS
        f.write('subject_num='+str(subject)+',IS='+str(isresult/40.0)+'\n')
 



