import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from ATMS_retrieval import ATMS, extract_id_from_string
from feature_extractor import MultiModalFeatureExtractor
from unet import Pipe, DiffusionPriorUNet
from dataset import EEGDataset
from diffusers import StableDiffusionXLPipeline
import open_clip
from diffusers.schedulers import DDPMScheduler
from PIL import Image
import random

# 配置参数
class Config:
    # 预训练模型路径 (使用train_unet=True的权重)
    model_path = "./outputs/multimodal_diffusion/unet_True/model_epoch_99.pth"
    
    # 数据路径
    data_path = "../../Data/Preprocessed_data_250Hz"
    subjects = ['sub-01']
    
    # 特征字典路径
    superpixel_base_path = '../../Data/processed_images'
    
    # Stable Diffusion 配置
    sd_model_path = "./models/sd_xl_base_1.0.safetensors"
    ip_adapter_weight_name = "ip-adapter_sdxl_vit-h.safetensors"
    
    # 生成参数
    batch_size = 1
    num_inference_steps = 30  
    guidance_scale = 5.0     
    height = 768        
    width = 768
    output_dir = "./generated_images"
    
    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载特征字典
def load_feature_dict(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Feature file not found: {file_path}")
    return torch.load(file_path)

# 模型加载器
class ModelLoader:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.load_models()
        
    def load_models(self):
        # 加载特征提取器
        self.feature_extractor = MultiModalFeatureExtractor().to(self.device)
        
        # 初始化扩散模型
        self.diffusion_models = {
            'image': self._create_pipe('image'),
            'text': self._create_pipe('text'),
            'superpixel_few': self._create_pipe('superpixel_few'),
            'superpixel_middle': self._create_pipe('superpixel_middle'),
            'superpixel_large': self._create_pipe('superpixel_large')
        }
        
        # 加载预训练权重 (train_unet=True)
        checkpoint = torch.load(self.config.model_path, map_location=self.device)
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
        self.diffusion_models['image'].diffusion_prior.load_state_dict(
            checkpoint['image_diffusion_state_dict'])
        self.diffusion_models['text'].diffusion_prior.load_state_dict(
            checkpoint['text_diffusion_state_dict'])
        self.diffusion_models['superpixel_few'].diffusion_prior.load_state_dict(
            checkpoint['superpixel_few_diffusion_state_dict'])
        self.diffusion_models['superpixel_middle'].diffusion_prior.load_state_dict(
            checkpoint['superpixel_middle_diffusion_state_dict'])
        self.diffusion_models['superpixel_large'].diffusion_prior.load_state_dict(
            checkpoint['superpixel_large_diffusion_state_dict'])
        
        # 设置模型为评估模式
        self.feature_extractor.eval()
        for model in self.diffusion_models.values():
            model.diffusion_prior.eval()
        
        print("Feature extractor and diffusion models loaded successfully.")
    
    def _create_pipe(self, modality):
        """创建指定模态的扩散管道"""
        pipe = Pipe(
            diffusion_prior=DiffusionPriorUNet(),
            scheduler=DDPMScheduler(),
            device=self.device,
            modality=modality
        )
        return pipe
    
    # 加载Stable Diffusion和IP-Adapter
    def load_generation_models(self):
        # 检查模型文件是否存在
        if not os.path.exists(self.config.sd_model_path):
            raise FileNotFoundError(f"Stable Diffusion model not found at: {self.config.sd_model_path}")
        
        # 加载图像编码器（使用open_clip）
        self.image_encoder, _, self.feature_extractor = open_clip.create_model_and_transforms(
            'ViT-H-14', 
            pretrained='laion2b_s32b_b79k', 
            precision='fp16', 
            device=self.device
        )

        # 加载Stable Diffusion
        self.sd_pipe = StableDiffusionXLPipeline.from_single_file(
            self.config.sd_model_path,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)
        
        # 加载IP-Adapter - 加载5个相同的IP-Adapter
        self.sd_pipe.load_ip_adapter(
            "h94/IP-Adapter", 
            subfolder="sdxl_models",
            weight_name=[self.config.ip_adapter_weight_name] * 5,  # 加载5个相同的IP-Adapter
            torch_dtype=torch.float16
        )
        
        print("Stable Diffusion and IP-Adapter loaded successfully from local files.")

# CLIP嵌入生成器
class CLIPEmbeddingGenerator:
    def __init__(self, model_loader, config):
        self.model_loader = model_loader
        self.config = config
        self.device = config.device
        
        # 加载超像素图特征字典（三个级别）
        self.superpixel_dicts = {}
        for level in ['few', 'middle', 'large']:
            self.superpixel_dicts[f'test_{level}'] = load_feature_dict(
                os.path.join(config.superpixel_base_path, f"test_images_{level}_slic.pt")
            )
    
    def _get_superpixel_features(self, img_paths, level, mode='test'):
        """获取指定级别的超像素特征"""
        features = []
        dict_key = f'{mode}_{level}'
        
        for img_path in img_paths:
            img_name = os.path.basename(img_path)
            features.append(self.superpixel_dicts[dict_key][img_name])
        
        return torch.stack(features).to(self.device)
    
    def generate_clip_embeddings(self, eeg_data, img_paths):
        """生成多模态CLIP嵌入"""
        with torch.no_grad():
            # 准备主题ID
            batch_size = eeg_data.size(0)
            sub = self.config.subjects[0]
            subject_id = extract_id_from_string(sub)
            subject_ids = torch.full((batch_size,), subject_id, dtype=torch.long).to(self.device)
            
            # 通过特征提取器获取条件嵌入
            cond_embeds = self.model_loader.feature_extractor.eeg_model(eeg_data, subject_ids).float()
            
            # 为每个模态生成CLIP嵌入（使用扩散模型）
            clip_embeddings = {}
            
            # 图像模态
            img_clip = self._diffusion_sample(
                self.model_loader.diffusion_models['image'], 
                cond_embeds
            )
            clip_embeddings['image'] = img_clip
            
            # 文本模态
            text_clip = self._diffusion_sample(
                self.model_loader.diffusion_models['text'], 
                cond_embeds
            )
            clip_embeddings['text'] = text_clip
            
            # 超像素模态 - 三个级别
            for level in ['few', 'middle', 'large']:
                modality_key = f'superpixel_{level}'
                clip = self._diffusion_sample(
                    self.model_loader.diffusion_models[modality_key], 
                    cond_embeds
                )
                clip_embeddings[modality_key] = clip
            
            return clip_embeddings
    
    def _diffusion_sample(self, diffusion_model, cond_embeds):
        """使用扩散模型生成嵌入（采样过程）"""
        # 初始化随机噪声
        noise = torch.randn_like(cond_embeds)
        
        # 设置扩散模型的调度器
        diffusion_model.scheduler.set_timesteps(self.config.num_inference_steps)
        
        # 初始噪声
        latents = noise
        
        # 扩散采样过程
        for t in diffusion_model.scheduler.timesteps:
            # 扩展时间步以匹配批量大小
            timesteps = torch.full((cond_embeds.shape[0],), t, device=self.device, dtype=torch.long)
            
            # 预测噪声
            noise_pred = diffusion_model.diffusion_prior(
                latents, 
                timesteps,
                cond_embeds
            )
            
            # 计算前一步的噪声更小的样本
            latents = diffusion_model.scheduler.step(
                noise_pred, t, latents
            ).prev_sample
        
        return latents

# 图像生成器
class ImageGenerator:
    def __init__(self, model_loader, config):
        self.model_loader = model_loader
        self.config = config
        self.device = config.device
        os.makedirs(config.output_dir, exist_ok=True)
        
        # 分层注入配置 - 定义不同模态在UNet中的注入位置
        self.hierarchical_scales = [
            # 图像模态 - 注入上采样后半部分
            {
                "down": {"block_0": 0.0, "block_1": 0.0, "block_2": 0.0, "block_3": 0.0},
                "up": {"block_0": 0.0, "block_1": 0.0, "block_2": 1.0, "block_3": 1.0}
            },
            # 文本模态 - 全域注入
            {
                "down": {"block_0": 0.25, "block_1": 0.25, "block_2": 0.25, "block_3": 0.25},
                "up": {"block_0": 0.25, "block_1": 0.25, "block_2": 0.25, "block_3": 0.25}
            },
            # few超像素 - 注入下采样前半部分
            {
                "down": {"block_0": 1.0, "block_1": 1.0, "block_2": 0.0, "block_3": 0.0},
                "up": {"block_0": 0.0, "block_1": 0.0, "block_2": 0.0, "block_3": 0.0}
            },
            # middle超像素 - 注入下采样后半部分
            {
                "down": {"block_0": 0.0, "block_1": 0.0, "block_2": 1.0, "block_3": 1.0},
                "up": {"block_0": 0.0, "block_1": 0.0, "block_2": 0.0, "block_3": 0.0}
            },
            # large超像素 - 注入上采样前半部分
            {
                "down": {"block_0": 0.0, "block_1": 0.0, "block_2": 0.0, "block_3": 0.0},
                "up": {"block_0": 1.0, "block_1": 1.0, "block_2": 0.0, "block_3": 0.0}
            }
        ]
        
        # 全局权重比例
        self.global_scales = [0.3, 0.3, 0.1, 0.1, 0.2]  # image, text, few, middle, large
    
    def prepare_ip_adapter_embeddings(self, embed):
        """准备IP-Adapter输入的嵌入格式"""
        embed = embed.to(torch.float16).to(self.device)
        uncond_embeds = torch.zeros_like(embed, dtype=embed.dtype, device=self.device)
        return torch.stack([uncond_embeds, embed], dim=0)
    
    def generate_images(self, clip_embeddings, prompts=None, separate_generate=False):
        """使用CLIP嵌入生成图像，支持分离生成和联合生成模式"""
        generated_images = []
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # 对每个样本生成图像
        for i in range(len(clip_embeddings['image'])):
            # 获取当前样本的CLIP嵌入
            image_emb = clip_embeddings['image'][i].unsqueeze(0)
            text_emb = clip_embeddings['text'][i].unsqueeze(0)
            superpixel_few_emb = clip_embeddings['superpixel_few'][i].unsqueeze(0)
            superpixel_middle_emb = clip_embeddings['superpixel_middle'][i].unsqueeze(0)
            superpixel_large_emb = clip_embeddings['superpixel_large'][i].unsqueeze(0)
            
            # 准备提示词
            if prompts is None:
                prompt = "a generated image"
            else:
                prompt = prompts[i]
            
            # 分离生成模式：每个模态单独生成图像
            if separate_generate:
                outputs = []
                modalities = ['image', 'text', 'superpixel_few', 'superpixel_middle', 'superpixel_large']
                
                # 在这里定义 embeddings，确保在变量定义之后
                embeddings = [image_emb, text_emb, superpixel_few_emb, superpixel_middle_emb, superpixel_large_emb]
                
                for j, (modality, emb) in enumerate(zip(modalities, embeddings)):
                    # 准备IP-Adapter输入 - 当前模态使用真实嵌入，其他模态使用零嵌入
                    ip_adapter_input_list = []
                    for k in range(5):
                        if k == j:
                            # 当前模态使用真实嵌入
                            ip_adapter_input_list.append(self.prepare_ip_adapter_embeddings(emb))
                        else:
                            # 其他模态使用零嵌入
                            zero_emb = torch.zeros_like(emb)
                            ip_adapter_input_list.append(self.prepare_ip_adapter_embeddings(zero_emb))
                    
                    # 创建包含5个配置的列表，当前模态使用真实配置，其他使用禁用配置
                    scale_configs = []
                    for k in range(5):
                        if k == j:
                            scale_configs.append(self.hierarchical_scales[j])
                        else:
                            # 创建禁用配置 - 所有block的scale为0
                            disable_config = {
                                "down": {"block_0": 0.0, "block_1": 0.0, "block_2": 0.0, "block_3": 0.0},
                                "up": {"block_0": 0.0, "block_1": 0.0, "block_2": 0.0, "block_3": 0.0}
                            }
                            scale_configs.append(disable_config)
                    
                    # 设置当前模态的分层注入配置
                    self.model_loader.sd_pipe.set_ip_adapter_scale(scale_configs)
                    
                    # 生成
                    with torch.no_grad():
                        image = self.model_loader.sd_pipe(
                            prompt=prompt,
                            ip_adapter_image_embeds=ip_adapter_input_list,
                            height=self.config.height,
                            width=self.config.width,
                            num_inference_steps=self.config.num_inference_steps,
                            guidance_scale=self.config.guidance_scale
                        ).images[0]
                    
                    # 保存图像
                    img_path = os.path.join(self.config.output_dir, f"generated_{i}_{modality}.png")
                    image.save(img_path)
                    outputs.append(image)
                    print(f"Generated {modality} image saved to {img_path}")
                
                generated_images.append(outputs)
            
            # 联合生成模式：所有模态一起生成图像
            else:
                # 准备所有模态的IP-Adapter输入
                ip_adapter_input = [
                    self.prepare_ip_adapter_embeddings(image_emb),
                    self.prepare_ip_adapter_embeddings(text_emb),
                    self.prepare_ip_adapter_embeddings(superpixel_few_emb),
                    self.prepare_ip_adapter_embeddings(superpixel_middle_emb),
                    self.prepare_ip_adapter_embeddings(superpixel_large_emb)
                ]
                
                # 方法1: 只使用分层配置
                # self.model_loader.sd_pipe.set_ip_adapter_scale(self.hierarchical_scales)
                
                # 方法2: 将全局权重应用到分层配置中
                weighted_hierarchical_scales = []
                for global_scale, config in zip(self.global_scales, self.hierarchical_scales):
                    weighted_config = {}
                    for direction in ['down', 'up']:
                        weighted_config[direction] = {}
                        for block_name, block_scale in config[direction].items():
                            # 将全局权重应用到每个块的缩放值
                            weighted_config[direction][block_name] = block_scale * global_scale
                    
                    weighted_hierarchical_scales.append(weighted_config)
                
                self.model_loader.sd_pipe.set_ip_adapter_scale(weighted_hierarchical_scales)
                
                # 生成图像
                with torch.no_grad():
                    image = self.model_loader.sd_pipe(
                        prompt=prompt,
                        ip_adapter_image_embeds=ip_adapter_input,
                        height=self.config.height,
                        width=self.config.width,
                        num_inference_steps=self.config.num_inference_steps,
                        guidance_scale=self.config.guidance_scale
                    ).images[0]
                
                # 保存图像
                img_path = os.path.join(self.config.output_dir, f"generated_{i}_all.png")
                image.save(img_path)
                generated_images.append(image)
                print(f"Generated fused image saved to {img_path}")
        
        return generated_images

# 渐进式模态融合生成器
class ProgressiveGenerator:
    def __init__(self, model_loader, config):
        self.model_loader = model_loader
        self.config = config
        self.device = config.device
        self.output_dir = os.path.join(config.output_dir, "progressive")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 定义模态组合序列
        self.modality_combinations = [
            {"name": "few_only", "modalities": ["superpixel_few"]},
            {"name": "few+middle", "modalities": ["superpixel_few", "superpixel_middle"]},
            {"name": "all_superpixels", "modalities": ["superpixel_few", "superpixel_middle", "superpixel_large"]},
            {"name": "all_superpixels+text", "modalities": ["superpixel_few", "superpixel_middle", "superpixel_large", "text"]},
            {"name": "all_modalities", "modalities": ["superpixel_few", "superpixel_middle", "superpixel_large", "text", "image"]}
        ]
        
        # 分层注入配置
        self.hierarchical_scales = {
            "image": {
                "down": {"block_0": 0.0, "block_1": 0.0, "block_2": 0.0, "block_3": 0.0},
                "up": {"block_0": 0.0, "block_1": 0.0, "block_2": 1.0, "block_3": 1.0}
            },
            "text": {
                "down": {"block_0": 0.25, "block_1": 0.25, "block_2": 0.25, "block_3": 0.25},
                "up": {"block_0": 0.25, "block_1": 0.25, "block_2": 0.25, "block_3": 0.25}
            },
            "superpixel_few": {
                "down": {"block_0": 1.0, "block_1": 1.0, "block_2": 0.0, "block_3": 0.0},
                "up": {"block_0": 0.0, "block_1": 0.0, "block_2": 0.0, "block_3": 0.0}
            },
            "superpixel_middle": {
                "down": {"block_0": 0.0, "block_1": 0.0, "block_2": 1.0, "block_3": 1.0},
                "up": {"block_0": 0.0, "block_1": 0.0, "block_2": 0.0, "block_3": 0.0}
            },
            "superpixel_large": {
                "down": {"block_0": 0.0, "block_1": 0.0, "block_2": 0.0, "block_3": 0.0},
                "up": {"block_0": 1.0, "block_1": 1.0, "block_2": 0.0, "block_3": 0.0}
            }
        }
        
        # 模态权重
        self.modality_weights = {
            "image": 0.3,
            "text": 0.3,
            "superpixel_few": 0.1,
            "superpixel_middle": 0.1,
            "superpixel_large": 0.2
        }
    
    def prepare_ip_adapter_embeddings(self, embed):
        """准备IP-Adapter输入的嵌入格式"""
        embed = embed.to(torch.float16).to(self.device)
        uncond_embeds = torch.zeros_like(embed, dtype=embed.dtype, device=self.device)
        return torch.stack([uncond_embeds, embed], dim=0)
    
    def generate_progressive_images(self, clip_embeddings, prompts=None):
        """生成渐进式模态融合的图像序列"""
        results = {}
        
        # 对每个样本生成图像序列
        for i in range(len(clip_embeddings['image'])):
            # 获取当前样本的CLIP嵌入
            image_emb = clip_embeddings['image'][i].unsqueeze(0)
            text_emb = clip_embeddings['text'][i].unsqueeze(0)
            superpixel_few_emb = clip_embeddings['superpixel_few'][i].unsqueeze(0)
            superpixel_middle_emb = clip_embeddings['superpixel_middle'][i].unsqueeze(0)
            superpixel_large_emb = clip_embeddings['superpixel_large'][i].unsqueeze(0)
            
            # 准备提示词
            if prompts is None:
                prompt = "a generated image"
            else:
                prompt = prompts[i]
            
            # 为每个模态组合生成图像
            sequence_images = []
            for combo in self.modality_combinations:
                # 准备IP-Adapter输入
                ip_adapter_input = []
                scale_configs = []
                
                # 为每个模态准备嵌入和配置
                for modality in ["image", "text", "superpixel_few", "superpixel_middle", "superpixel_large"]:
                    if modality in combo["modalities"]:
                        # 使用真实嵌入
                        if modality == "image":
                            ip_adapter_input.append(self.prepare_ip_adapter_embeddings(image_emb))
                        elif modality == "text":
                            ip_adapter_input.append(self.prepare_ip_adapter_embeddings(text_emb))
                        elif modality == "superpixel_few":
                            ip_adapter_input.append(self.prepare_ip_adapter_embeddings(superpixel_few_emb))
                        elif modality == "superpixel_middle":
                            ip_adapter_input.append(self.prepare_ip_adapter_embeddings(superpixel_middle_emb))
                        elif modality == "superpixel_large":
                            ip_adapter_input.append(self.prepare_ip_adapter_embeddings(superpixel_large_emb))
                        
                        # 应用权重到分层配置
                        weighted_config = {}
                        for direction in ['down', 'up']:
                            weighted_config[direction] = {}
                            for block_name, block_scale in self.hierarchical_scales[modality][direction].items():
                                weighted_config[direction][block_name] = block_scale * self.modality_weights[modality]
                        scale_configs.append(weighted_config)
                    else:
                        # 使用零嵌入
                        zero_emb = torch.zeros_like(image_emb)
                        ip_adapter_input.append(self.prepare_ip_adapter_embeddings(zero_emb))
                        
                        # 禁用配置
                        disable_config = {
                            "down": {"block_0": 0.0, "block_1": 0.0, "block_2": 0.0, "block_3": 0.0},
                            "up": {"block_0": 0.0, "block_1": 0.0, "block_2": 0.0, "block_3": 0.0}
                        }
                        scale_configs.append(disable_config)
                
                # 设置分层注入配置
                self.model_loader.sd_pipe.set_ip_adapter_scale(scale_configs)
                
                # 生成图像
                with torch.no_grad():
                    image = self.model_loader.sd_pipe(
                        prompt=prompt,
                        ip_adapter_image_embeds=ip_adapter_input,
                        height=self.config.height,
                        width=self.config.width,
                        num_inference_steps=self.config.num_inference_steps,
                        guidance_scale=self.config.guidance_scale
                    ).images[0]
                
                # 保存图像
                img_path = os.path.join(self.output_dir, f"generated_{i}_{combo['name']}.png")
                image.save(img_path)
                sequence_images.append(image)
                print(f"Generated {combo['name']} image for sample {i} saved to {img_path}")
            
            results[i] = {
                "images": sequence_images,
                "combinations": [combo["name"] for combo in self.modality_combinations]
            }
        
        return results

# 主函数
def main():
    # 初始化配置
    config = Config()
    
    # 加载模型
    model_loader = ModelLoader(config)
    model_loader.load_generation_models()
    
    # 准备嵌入生成器和图像生成器
    clip_generator = CLIPEmbeddingGenerator(model_loader, config)
    image_generator = ImageGenerator(model_loader, config)
    progressive_generator = ProgressiveGenerator(model_loader, config)
    
    # 加载测试数据集
    test_dataset = EEGDataset(config.data_path, subjects=config.subjects, train=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # 处理测试数据
    for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(test_loader):
        eeg_data = eeg_data.to(config.device)
        
        # 生成CLIP嵌入
        clip_embeddings = clip_generator.generate_clip_embeddings(eeg_data, img)
        
        # 生成图像 - 分离模式和联合模式
        separated_images = image_generator.generate_images(
            clip_embeddings, text, separate_generate=True
        )
        
        fused_image = image_generator.generate_images(
            clip_embeddings, text, separate_generate=False
        )

        # 生成渐进式模态融合图像序列
        progressive_results = progressive_generator.generate_progressive_images(
            clip_embeddings, text
        )
        
        # 只处理一个batch作为示例
        break

if __name__ == "__main__":
    main()