from collections import defaultdict
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset

import SimpleITK as sitk
# from timm.models.layers import to_3tuple
import numpy as np
import random
from scipy import ndimage

def random_crop(image, crop_shape=224):
    _, _, z_shape, y_shape, x_shape = image.shape
    z_min = np.random.randint(0, z_shape - crop_shape)
    y_min = np.random.randint(0, y_shape - crop_shape)
    x_min = np.random.randint(0, x_shape - crop_shape)
    image = image[:, :, z_min:z_min+crop_shape, y_min:y_min+crop_shape, x_min:x_min+crop_shape]
    return image

def center_crop(image, target_shape=224):
    _, _, z_shape, y_shape, x_shape = image.shape
    z_min = z_shape // 2 - target_shape // 2
    y_min = y_shape // 2 - target_shape // 2
    x_min = x_shape // 2 - target_shape // 2
    image = image[:, :, z_min:z_min+target_shape, y_min:y_min+target_shape, x_min:x_min+target_shape]
    return image

def pad_img(img, spacing, size=224):
    '''pad img to square.
    '''
    x, y, z = img.shape

    x *= spacing[2]
    y *= spacing[1]
    z *= spacing[0]

    # print ('input image: ', x, y, z, spacing)

    img = img.unsqueeze(0).unsqueeze(0) # BCHWD
    max_size = max(x, y, z)
    new_size = (int(size*x/max_size), int(size*y/max_size), int(size*z/max_size))
    img = F.interpolate(img,size=new_size,mode='trilinear',align_corners=True)

    x,y,z = new_size
    new_im = torch.zeros((1,1,size,size,size))
    x_min = int((size - x) / 2)
    x_max = x_min + x
    y_min = int((size - y) / 2)
    y_max = y_min + y
    z_min = int((size - z) / 2)
    z_max = z_min + z
    new_im[:,:,x_min:x_max,y_min:y_max,z_min:z_max] = img
    
    return new_im
    
def norm_img(img):
    return (img - img.min())/(img.max() - img.min())

def randomflip_x(image, p=0.5):
    if random.random() > p:
        return image
    else:
        image_arr = image.numpy()
        image_arr = image_arr[:, :, :, :, ::-1]
        return torch.Tensor(image_arr.copy())

def randomflip_y(image, p=0.5):
    if random.random() > p:
        return image
    else:
        image_arr = image.numpy()
        image_arr = image_arr[:, :, :, ::-1, :]
        return torch.Tensor(image_arr.copy())

def randomflip_z(image, p=0.5):
    if random.random() > p:
        return image
    else:
        image_arr = image.numpy()
        image_arr = image_arr[:, :, ::-1, :, :]
        return torch.Tensor(image_arr.copy())

def random_flip(image, mode='x', p=0.5):
    if mode == 'x':
        image = randomflip_x(image, p=p)
    elif mode == 'y':
        image = randomflip_y(image, p=p)
    elif mode == 'z':
        image = randomflip_z(image, p=p)
    else:
        raise NotImplementedError(f'Unknown flip mode ({mode})')
    return image

def rotate(image, p=0.5):
    if random.random() > p:
        return image
    else:
        angle = random.randint(-10, 10)
        r_image = ndimage.rotate(image, angle=angle, axes=(-2, -1), reshape=True)
        if r_image.shape != image.shape:
            r_image = center_crop(r_image)
        return torch.Tensor(r_image)

def RandScaleIntensity(image, factor=0.2, p=0.5):
    if random.random() > p:
        return image
    else:
        scale = 1.0 + (random.random() * 2 * factor - factor)
        image = image * scale
        return image

class BrainSCKTrainDataset(Dataset):

    def __init__(self, datalist, dataset_type) -> None:
        super().__init__()
        # imgpath, report
        df_list = []

        with open(datalist) as f:
            lines = f.readlines()
            for line in lines:
                imgpath,report = line.strip('\n').strip('"').split('\t')
                imgpath = imgpath.replace('', '')
                df_list.append((imgpath, report, dataset_type))        

        print ('df_list: ', len(df_list), df_list[0])
        self.df = df_list
        

    def __getitem__(self, index):
        imgpath,report,tag = self.df[index]
        img_nii = sitk.ReadImage(imgpath)
        spacing = img_nii.GetSpacing()
        img = torch.Tensor(sitk.GetArrayFromImage(img_nii).astype(float))
        # img = sitk.GetArrayFromImage(img_nii).astype(float)
        # print (imgpath, img.shape, spacing)
        img = norm_img(img)
        img = pad_img(img, spacing, size=224)
        
        return img, report, tag

    def __len__(self):
        return len(self.df)


class BrainSCKTrainCollator:
    def __init__(self):
        return
    def __call__(self, batch):
        inputs = defaultdict(list)
        for data in batch:
            inputs['images'].append(data[0])
            inputs['reports'].append(data[1])
            inputs['tag'].append(data[2])

        inputs['images'] = torch.cat(inputs['images'], 0)

        return inputs

class BrainSCKValidDataset(Dataset):
    def __init__(self, datalist) -> None:
        super().__init__()

        df_list = []
        # filename = f'./local_data/{datalist}.csv'
        filename = datalist
        print('load data from', filename)
        with open(filename) as f:
            lines = f.readlines()
            for line in lines:
                imgpath,report = line.strip('\n').strip('"').split('\t')
                imgpath = imgpath.replace('', '')
                df_list.append((imgpath,report))
        
        print ('df_list: ', len(df_list), df_list[0])
        self.df = df_list

    def __getitem__(self, index):
        imgpath,report = self.df[index]
        img_nii = sitk.ReadImage(imgpath)
        spacing = img_nii.GetSpacing()
        img = torch.Tensor(sitk.GetArrayFromImage(img_nii).astype(float))
        # print (imgpath, img.shape, spacing)
        img = norm_img(img)
        img = pad_img(img, spacing, size=224)
        # img = center_crop(img, target_shape=224)

        return img, report

    def __len__(self):
        return len(self.df)

class BrainSCKValidCollator:
    def __init__(self):
        return
    
    def __call__(self, batch):
        inputs = defaultdict(list)
        for data in batch:
            inputs['images'].append(data[0])
            inputs['reports'].append(data[1])

        inputs['images'] = torch.cat(inputs['images'], 0)

        return inputs

