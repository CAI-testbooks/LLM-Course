from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
import os

import argparse
parser=argparse.ArgumentParser(description="Template")
parser.add_argument('--targetpath',default='Data/EEGDataset/train_images/')
parser.add_argument('--gen_img_path',default='generated_images')


opt = parser.parse_args()

targetpath=opt.targetpath
gen_img_path=opt.gen_img_path


def load_and_preprocess_image(image_path):
    image=Image.open(image_path)
    image = image.convert('RGB')
    image=image.resize((224,224))
    image = np.array(image)
    return image


def ssim_img_score (img1_path,img2_path):
    # Load the two images and preprocess them for CLIP
    img1 = load_and_preprocess_image(img1_path)
    img2 = load_and_preprocess_image(img2_path)
    
    result=ssim(img1, img2, multichannel=True,data_range=255.0)
    return result



catfile=open('Data/EmbeddingFiles/cat.txt')
catlist=catfile.read().split('\n')
imgfile=open('Data/EmbeddingFiles/image_cat.txt')
imglist=imgfile.read().split('\n')

def get_score(subnum:int):
    ssim_score=0
    k=0
    gen_img_path=opt.gen_img_path
    subjectpath=os.path.join(gen_img_path,str(subnum))
    for catidx in range(len(catlist)): 
        catpath=os.path.join(subjectpath,str(catidx))
        if not os.path.isdir(catpath):
            continue
        for imgidx in os.listdir(catpath):
            for imgname in range(7):
                target_img_path=os.path.join(targetpath,catlist[catidx],imglist[int(imgidx)]+'.JPEG')
                
                # target_img_path=('/opt/data/private/EEG_imagenet_subjects/generator_gt/input_photolabels/'+str(catidx)+'.jpg')
                gen_img_path=os.path.join(subjectpath,str(catidx),imgidx,str(imgname)+'.jpg')
                if not os.path.exists(gen_img_path):
                    continue
                ssim_score+=ssim_img_score(target_img_path,gen_img_path)
                k+=1
    return ssim_score,k



if __name__ == "__main__":
    ssim_score=0
    k=0
    f=open('metrics/SSIMresult.txt','a')
    f.write(gen_img_path+'\n')

    for sub in range(1,7):
        sc, num = get_score(sub)  #give the path to two images.
        print(str(sub)+':ssim_score='+str(sc)+',num='+str(num)+',avg='+str(sc/num)+'\n')
        f.write(str(sub)+':ssim_score='+str(sc)+',num='+str(num)+',avg='+str(sc/num)+'\n')
        ssim_score+=sc
        k+=num
    
    print(ssim_score/k)
    f.write('total_ssim_score='+str(ssim_score/k)+'\n')

