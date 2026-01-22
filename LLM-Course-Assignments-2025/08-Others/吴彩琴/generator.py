import os,sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.append(os.getcwd())
import torch
from PIL import Image
import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
from eeg_models.DATASET import EEGDataset, Splitter
from eeg_models.EEGNet import VisModel
from eeg_models.EEGNet import Model as EEGModel

from torchvision import transforms as trn
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from glide_text2im.download import load_checkpoint
import os
import argparse
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--subject',type=int,default=1)
parser.add_argument('--eeg_epoch',type=int,default=100)
parser.add_argument('--cls_decay',type=str,default='0.001')
parser.add_argument('--cls_epoch',type=int,default=200)
parser.add_argument('--sp',default='test')
parser.add_argument('--mlp_epoch',type=int,default=250)
parser.add_argument('--vis_epoch',type=int,default=200)
# parser.add_argument()
opt=parser.parse_args()

pretrained_file='saved_models/EEG_sem_classification/subject_'+str(opt.subject)+'/decay_'+opt.cls_decay+'/epoch_'+str(opt.cls_epoch)+'.pth'
mlpfile='saved_models/mlpmodels/subject_'+str(opt.subject)+'/mlp_epoch_'+str(opt.mlp_epoch)+'.pth'
visfile='saved_models/EEG_vis_clip/subject_'+str(opt.subject)+'/epoch_'+str(opt.vis_epoch)+'.pth'

saved_path='generated_images/'
splits_path='Data/EEGDataset/block_splits_by_image_subject_'+str(opt.subject)+'.pth' 

vismodel = VisModel().cuda()
vismodel.load_state_dict(torch.load(visfile))
vismodel.eval()
eegmodel = EEGModel(embsize=160).cuda()
eegmodel.load_state_dict(torch.load(pretrained_file))
eegmodel.eval()

mlp_model=nn.Sequential(
    nn.Linear(160,500),
    nn.ReLU(),
    nn.Linear(500,6*512)
).cuda()
mlp_model.load_state_dict(torch.load(mlpfile,map_location='cuda'))

preemb=torch.load('Data/EmbeddingFiles/prefix.pth').cuda()
guidance_scale=7.0
mlp_model.eval()

def show_images(batch: th.Tensor):
    """ Display a batch of images inline. """
    scaled = ((batch + 1)*127.5).round().clamp(0,255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    img=Image.fromarray(reshaped.numpy())
    return img

tensor2init=trn.Compose([
	trn.Resize((64,64)),
	trn.ToTensor()
])


dataset = EEGDataset("Data/EEGDataset/eeg_55_95_std.pth")
loaders = {split: DataLoader(Splitter(dataset, split_path=splits_path, split_num=0, split_name=split),
                             batch_size=1, drop_last=False, shuffle=True) for split in
           ["test"]}



def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + th.zeros(broadcast_shape, device=timesteps.device)

def stochastic_encode(model, x0, t=torch.Tensor([80]).long()):
    with torch.no_grad():
        sqrt_alphas_cumprod = model.sqrt_alphas_cumprod
        sqrt_one_minus_alphas_cumprod = model.sqrt_one_minus_alphas_cumprod
        noise = torch.randn_like(x0)
        return (_extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                _extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)

has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda')
options = model_and_diffusion_defaults()
options['use_fp16'] = has_cuda
options['timestep_respacing'] = '100' # use 100 diffusion steps for fast sampling
model, diffusion = create_model_and_diffusion(**options)
model.eval()

if has_cuda:
    model.convert_to_fp16()
model.to(device)
# model.load_state_dict(th.load('net_models/base.pt'))
model.load_state_dict(load_checkpoint('base', device))
print('total base parameters', sum(x.numel() for x in model.parameters()))

batch_size=1
guidance_scale=7.0

train_latent=torch.load('Data/EmbeddingFiles/pca_20')
print(train_latent.shape)
latent_tensor_norm=train_latent/torch.norm(train_latent,dim=1,keepdim=True)
train_imgidx_list=open('Data/EmbeddingFiles/train_lantentidx.txt').read().split('\n')
latent_tensor_norm=torch.stack([latent_tensor_norm[int(train_imgidx_list[i])] for i in range(len(train_imgidx_list))],dim=0)
imgidx_list=open('Data/EmbeddingFiles/image_cat.txt').read().split('\n')
catidx_list=open('Data/EmbeddingFiles/cat.txt').read().split('\n')
masktensor=torch.load('Data/EmbeddingFiles/mask.pth').cuda()

with torch.no_grad():
    for split in [opt.sp]:
        for i, (input_eeg, target_label, idx, subnum, imgnum) in enumerate(loaders[split]):
            for j in range(7):
                input=input_eeg[:,:,:,0+40*j:160+40*j]
                input=input.to('cuda')
                eeg_feature,output=eegmodel(input)
                _, output_label = output.data.max(1)
                glide_feature=mlp_model(eeg_feature)
                batsize=glide_feature.shape[0]
                embtensor=glide_feature.reshape((batsize,6,512))
                inputvis=input[:,:,[28,29,30,59,60,61,62,63,91,92,93,94,125,126]]
                vis_feature=vismodel(inputvis).detach().cpu()
                for b in range(batsize):
                    #create path
                    path=os.path.join(saved_path,str(subnum[b].item()),str(target_label[b].item()),str(imgnum[b].item()))
                    if(os.path.exists(path)==0): os.makedirs(path)
                    imgpath=os.path.join(path,str(j)+'.jpg')

                    #selecting init images
                    outputimg=vis_feature/torch.norm(vis_feature,dim=1,keepdim=True)
                    mat=torch.mm(outputimg,latent_tensor_norm.T)
                    a=mat.flatten(start_dim=0).tolist()
                    sorted_id = sorted(range(len(a)), key=lambda k: a[k], reverse=True)
                    cat_tar=catidx_list[output_label]
                    cat_img=''
                    for m in sorted_id:
                        image_idx=train_imgidx_list[m]
                        cat_img=imgidx_list[int(image_idx)]
                        if(cat_img.split('_')[0]==cat_tar):
                            break
                    image_init_path='Data/EEGDataset/train_images/'+cat_img.split('_')[0]+'/'+cat_img+'.JPEG'
                    image_init=Image.open(image_init_path).convert('RGB')
                    img_init=tensor2init(image_init).unsqueeze(0)
                    img_init=img_init*2-1
                    img_init=stochastic_encode(model=diffusion,x0=img_init)

                    

                    guidance_scale = 7.0
                    contokens=th.cat((preemb[0:3],embtensor[b],masktensor[9:]),dim=0)
                    
                    tokens=th.cat((contokens.unsqueeze(0),masktensor.unsqueeze(0)),dim=0)
                    # tokens=th.cat((masktensor.unsqueeze(0),masktensor.unsqueeze(0)),dim=0)
                    model_kwargs = dict(
                        tokens=tokens,
                        mask=None
                    )
                    # Create a classifier-free guidance sampling function
                    def model_fn(x_t, ts, **kwargs):
                        half = x_t[: len(x_t) // 2]
                        combined = th.cat([half, half], dim=0)
                        model_out = model(combined, ts, **kwargs)
                        eps, rest = model_out[:, :3], model_out[:, 3:]
                        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
                        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                        eps = th.cat([half_eps, half_eps], dim=0)
                        return th.cat([eps, rest], dim=1)
                    
                    noise_init=th.cat([img_init.cuda(),th.randn_like(img_init,device='cuda')],dim=0)
                    model.del_cache()
                    samples = diffusion.p_sample_loop(
                        model_fn,
                        (2, 3, options["image_size"], options["image_size"]),
                        device=device,
                        clip_denoised=True,
                        progress=True,
                        model_kwargs=model_kwargs,
                        cond_fn=None,
                        noise=noise_init
                    )[:1]
                    model.del_cache()
                    img=show_images(samples)
                    img.save(imgpath)



