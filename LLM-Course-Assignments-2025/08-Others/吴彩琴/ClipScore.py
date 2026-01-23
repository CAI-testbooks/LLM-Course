import torch
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer
from PIL import Image
import os
import cv2

import argparse
parser=argparse.ArgumentParser(description="Template")
parser.add_argument('--targetpath',default='Data/EEGDataset/train_images/')
parser.add_argument('--gen_img_path',default='generated_images/')
opt = parser.parse_args()

targetpath=opt.targetpath

# Load the CLIP model
# model_ID = "net_models/clip-vit-base-patch16"
model_ID = "openai/clip-vit-base-patch16"
model = CLIPModel.from_pretrained(model_ID)
preprocess = CLIPImageProcessor.from_pretrained(model_ID)


# Define a function to load an image and preprocess it for CLIP
def load_and_preprocess_image(image_path):
    # Load the image from the specified path
    image = Image.open(image_path)
    # Apply the CLIP preprocessing to the image
    image = preprocess(image, return_tensors="pt")
    # Return the preprocessed image
    return image


def clip_img_score (img1_path,img2_path):
    # Load the two images and preprocess them for CLIP
    image_a = load_and_preprocess_image(img1_path)["pixel_values"]
    image_b = load_and_preprocess_image(img2_path)["pixel_values"]

    # Calculate the embeddings for the images using the CLIP model
    with torch.no_grad():
        embedding_a = model.get_image_features(image_a)
        embedding_b = model.get_image_features(image_b)

    # Calculate the cosine similarity between the embeddings
    similarity_score = torch.nn.functional.cosine_similarity(embedding_a, embedding_b)
    return similarity_score.item()

catfile=open('Data/EmbeddingFiles/cat.txt')
catlist=catfile.read().split('\n')
#['n02389026', 'n03888257', 'n03584829']
imgfile=open('Data/EmbeddingFiles/image_cat.txt')
imglist=imgfile.read().split('\n')

def get_score(subnum:int):
    clips=0
    k=0
    gen_img_path=opt.gen_img_path
    subjectpath=os.path.join(gen_img_path,str(subnum))
    for catidx in range(len(catlist)): 
        catpath=os.path.join(subjectpath,str(catidx))
        if not os.path.exists(catpath):
            continue
        for imgidx in os.listdir(catpath):
            for imgname in range(7):
                target_img_path=os.path.join(targetpath,catlist[catidx],imglist[int(imgidx)]+'.JPEG')
                gen_img_path=os.path.join(subjectpath,str(catidx),imgidx,str(imgname)+'.jpg')
                if not os.path.exists(gen_img_path):
                    continue
                clips+=clip_img_score(target_img_path,gen_img_path)
                k+=1
    return clips,k

if __name__=='__main__':
    file=open('metrics/clipscore.txt','a')
    clips=0
    k=0
    gen_img_path=opt.gen_img_path
    file.write(gen_img_path+'\n')
    for sub in [1,2,3,4,5,6]:
        print(sub)
        clip, num = get_score(sub)  #give the path to two images.
        print('clipscore=',clip,'num=',num)
        file.write(str(sub)+'\nclipscore='+str(clip/num)+',num='+str(num)+'\n')
        clips+=clip
        k+=num
    
    print(clips/k)
    file.write('total'+str(clips/k)+'\n')



