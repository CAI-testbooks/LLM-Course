import os
import torch
import torch.optim as optim
from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import datetime
import numpy as np
from ATMS_retrieval import ATMS, extract_id_from_string
from unet import Pipe, DiffusionPriorUNet
from dataset import EEGDataset
from diffusers.schedulers import DDPMScheduler
from loss import ClipLoss
import wandb

# 加载特征字典
def load_feature_dict(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Feature file not found: {file_path}")
    return torch.load(file_path)

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha  # MSE权重
        self.beta = beta    # 余弦相似度权重
        self.mse = nn.MSELoss()
        
    def forward(self, predictions, targets):
        mse_loss = self.mse(predictions, targets)
        
        # 计算余弦相似度损失
        cos_sim = F.cosine_similarity(predictions, targets, dim=-1)
        cos_loss = 1 - cos_sim.mean()  # 将相似度转换为损失
        
        # 组合损失
        combined_loss = self.alpha * mse_loss + self.beta * cos_loss
        
        return combined_loss, mse_loss, cos_loss

class MultimodalDiffusionTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
        
        # 加载超像素图特征字典（三个级别）
        self.superpixel_dicts = {}
        for level in ['few', 'middle', 'large']:
            self.superpixel_dicts[f'train_{level}'] = load_feature_dict(
                os.path.join(args.superpixel_base_path, f"training_images_{level}_slic.pt")
            )
            self.superpixel_dicts[f'test_{level}'] = load_feature_dict(
                os.path.join(args.superpixel_base_path, f"test_images_{level}_slic.pt")
            )
        
        # 初始化模型
        self.eeg_model = ATMS().to(self.device)
        
        # 初始化扩散模型
        self.diffusion_models = {
            'image': self._create_pipe('image'),
            'text': self._create_pipe('text'),
            'superpixel_few': self._create_pipe('superpixel_few'),
            'superpixel_middle': self._create_pipe('superpixel_middle'),
            'superpixel_large': self._create_pipe('superpixel_large')
        }
        
        # 初始化优化器
        all_params = list(self.eeg_model.parameters())
        for model in self.diffusion_models.values():
            all_params += list(model.diffusion_prior.parameters())
        
        self.optimizer = optim.Adam(all_params, lr=args.lr_diff)
        self.clip_loss = ClipLoss()
        self.combined_loss = CombinedLoss(alpha=0.7, beta=0.3)
        
        # 定义模态顺序 (低级特征在前)
        self.modalities = ['superpixel_few', 'superpixel_middle', 'superpixel_large', 'image', 'text']
        
        # 创建输出目录
        self.output_dir = args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化WandB
        wandb.init(project=args.project, entity=args.entity, name=args.name, config=args)
    
    def _create_pipe(self, modality):
        """创建指定模态的扩散管道"""
        pipe = Pipe(
            diffusion_prior=DiffusionPriorUNet(),
            scheduler=DDPMScheduler(),
            device=self.device,
            modality=modality
        )
        return pipe
    
    def _get_superpixel_features(self, img_paths, level, mode='train'):
        """获取指定级别的超像素特征"""
        features = []
        dict_key = f'{mode}_{level}'
        
        for img_path in img_paths:
            img_name = os.path.basename(img_path)
            features.append(self.superpixel_dicts[dict_key][img_name])
        
        return torch.stack(features).to(self.device)

    def _diffusion_step(self, diffusion_model, target_features, cond_embeds):
        """执行扩散模型的一个训练步骤，返回组合损失"""
        # 确保目标特征是3D [batch, seq_len, features]
        if target_features.dim() == 2:
            target_features = target_features.unsqueeze(1)
        
        # 1. 采样随机噪声
        noise = torch.randn_like(target_features)
        
        # 2. 采样随机时间步
        timesteps = torch.randint(
            0, diffusion_model.scheduler.config.num_train_timesteps,
            (target_features.shape[0],), device=self.device
        ).long()
        
        # 3. 添加噪声到目标特征
        noisy_features = diffusion_model.scheduler.add_noise(
            target_features, noise, timesteps
        )
        
        # 4. 预测噪声
        noise_pred = diffusion_model.diffusion_prior(
            noisy_features, timesteps, cond_embeds
        )
        
        # 5. 确保噪声预测和噪声形状一致
        if noise_pred.shape != noise.shape:
            # 如果序列长度不同但特征维度相同，扩展噪声
            if noise_pred.size(-1) == noise.size(-1) and noise.size(1) == 1:
                noise = noise.expand_as(noise_pred)
            else:
                raise ValueError(f"Shape mismatch: noise_pred {noise_pred.shape}, noise {noise.shape}")
        
        # 6. 计算组合损失
        combined_loss, mse_loss, cos_loss = self.combined_loss(noise_pred, noise)
        
        return combined_loss, mse_loss, cos_loss
    
    def train(self):
        # 加载数据集
        train_dataset = EEGDataset(self.args.data_path, subjects=self.args.subjects, train=True)
        test_dataset = EEGDataset(self.args.data_path, subjects=self.args.subjects, train=False)
        
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, 
                                 shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, 
                                shuffle=False, num_workers=4, pin_memory=True)
        
        # 训练循环
        for epoch in range(self.args.epochs):
            print(f"Epoch {epoch}")
            
            # 训练模式
            self.eeg_model.train()
            for model in self.diffusion_models.values():
                model.diffusion_prior.train()
            
            total_loss = 0
            multimodal_losses = {modality: 0 for modality in self.modalities}
            multimodal_mse = {modality: 0 for modality in self.modalities}
            multimodal_cos = {modality: 0 for modality in self.modalities}
            
            for batch_idx, batch_data in enumerate(train_loader):
                # 解包batch数据
                eeg_data, labels, text, text_features, img, img_features = batch_data
                
                # 移动到设备
                eeg_data = eeg_data.to(self.device)
                text_features = text_features.to(self.device)
                img_features = img_features.to(self.device)
                
                batch_size = eeg_data.size(0) 
                sub = self.args.subjects[0]
                subject_id = extract_id_from_string(sub)
                subject_ids = torch.full((batch_size,), subject_id, dtype=torch.long).to(self.device)
                
                # 通过EEG模型获取条件嵌入
                cond_embeds = self.eeg_model(eeg_data, subject_ids).float()
                
                # 为每个模态计算损失
                losses = {}
                mse_losses = {}
                cos_losses = {}
                
                # 图像模态
                losses['image'], mse_losses['image'], cos_losses['image'] = self._diffusion_step(
                    self.diffusion_models['image'], img_features, cond_embeds
                )

                # 文本模态
                losses['text'], mse_losses['text'], cos_losses['text'] = self._diffusion_step(
                    self.diffusion_models['text'], text_features, cond_embeds
                )
    
                # 超像素模态 (三个级别)
                for level in ['few', 'middle', 'large']:
                    modality_key = f'superpixel_{level}'
                    # 获取当前级别的超像素特征
                    sp_features = self._get_superpixel_features(img, level, mode='train')
                    
                    losses[modality_key], mse_losses[modality_key], cos_losses[modality_key] = self._diffusion_step(
                        self.diffusion_models[modality_key], sp_features, cond_embeds
                    )
                
                # 总损失（加权和）
                total_batch_loss = (
                    0.3 * losses['image'] + 
                    0.3 * losses['text'] + 
                    0.1 * losses['superpixel_few'] + 
                    0.1 * losses['superpixel_middle'] + 
                    0.2 * losses['superpixel_large']
                )
                
                # 反向传播
                self.optimizer.zero_grad()
                total_batch_loss.backward()
                self.optimizer.step()
                
                # 记录损失
                total_loss += total_batch_loss.item()
                for modality in losses:
                    multimodal_losses[modality] += losses[modality].item()
                    multimodal_mse[modality] += mse_losses[modality].item()
                    multimodal_cos[modality] += cos_losses[modality].item()
            
            # 计算平均损失
            avg_total_loss = total_loss / len(train_loader)
            avg_losses = {modality: loss / len(train_loader) for modality, loss in multimodal_losses.items()}
            avg_mse = {modality: loss / len(train_loader) for modality, loss in multimodal_mse.items()}
            avg_cos = {modality: loss / len(train_loader) for modality, loss in multimodal_cos.items()}
            
            # 评估
            if epoch % 5 == 0:
                val_loss, val_losses, val_mse, val_cos = self.evaluate(test_loader)
                
                # 记录到WandB
                log_data = {
                    "epoch": epoch,
                    "train/total_loss": avg_total_loss,
                    "val/total_loss": val_loss,
                }
                
                # 添加各模态损失
                for modality in avg_losses:
                    log_data[f"train/{modality}_loss"] = avg_losses[modality]
                    log_data[f"train/{modality}_mse"] = avg_mse[modality]
                    log_data[f"train/{modality}_cos"] = avg_cos[modality]
                    log_data[f"val/{modality}_loss"] = val_losses[modality]
                    log_data[f"val/{modality}_mse"] = val_mse[modality]
                    log_data[f"val/{modality}_cos"] = val_cos[modality]
                
                wandb.log(log_data)
                
                print(f"Epoch {epoch}/{self.args.epochs} | Train Loss: {avg_total_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # 保存模型
            if epoch % 20 == 0 or epoch == self.args.epochs - 1:
                self.save_model(epoch)
        
        wandb.finish()
    
    def evaluate(self, test_loader):
        self.eeg_model.eval()
        for model in self.diffusion_models.values():
            model.diffusion_prior.eval()
        
        val_loss = 0
        val_losses = {modality: 0 for modality in self.modalities}
        val_mse = {modality: 0 for modality in self.modalities}
        val_cos = {modality: 0 for modality in self.modalities}
        
        with torch.no_grad():
            for batch_idx, (eeg_data, labels, text, text_features, img, img_features) in enumerate(test_loader):
                # 移动到设备
                eeg_data = eeg_data.to(self.device)
                text_features = text_features.to(self.device)
                img_features = img_features.to(self.device)
                
                batch_size = eeg_data.size(0) 
                sub = self.args.subjects[0]
                subject_id = extract_id_from_string(sub)
                subject_ids = torch.full((batch_size,), subject_id, dtype=torch.long).to(self.device)
                
                # 通过EEG模型获取条件嵌入
                cond_embeds = self.eeg_model(eeg_data, subject_ids).float()
                
                # 为每个模态计算损失
                losses = {}
                mse_losses = {}
                cos_losses = {}
                
                # 图像模态
                losses['image'], mse_losses['image'], cos_losses['image'] = self._diffusion_step(
                    self.diffusion_models['image'], img_features, cond_embeds
                )
                
                # 文本模态
                losses['text'], mse_losses['text'], cos_losses['text'] = self._diffusion_step(
                    self.diffusion_models['text'], text_features, cond_embeds
                )
                
                # 超像素模态 (三个级别)
                for level in ['few', 'middle', 'large']:
                    modality_key = f'superpixel_{level}'
                    # 获取当前级别的超像素特征
                    sp_features = self._get_superpixel_features(img, level, mode='test')
                    
                    losses[modality_key], mse_losses[modality_key], cos_losses[modality_key] = self._diffusion_step(
                        self.diffusion_models[modality_key], sp_features, cond_embeds
                    )
                
                # 总损失
                total_batch_loss = (
                    0.3 * losses['image'] + 
                    0.3 * losses['text'] + 
                    0.1 * losses['superpixel_few'] + 
                    0.1 * losses['superpixel_middle'] + 
                    0.2 * losses['superpixel_large']
                )
                
                # 记录损失
                val_loss += total_batch_loss.item()
                for modality in losses:
                    val_losses[modality] += losses[modality].item()
                    val_mse[modality] += mse_losses[modality].item()
                    val_cos[modality] += cos_losses[modality].item()
        
        # 计算平均验证损失
        avg_val_loss = val_loss / len(test_loader)
        avg_val_losses = {modality: loss / len(test_loader) for modality, loss in val_losses.items()}
        avg_val_mse = {modality: loss / len(test_loader) for modality, loss in val_mse.items()}
        avg_val_cos = {modality: loss / len(test_loader) for modality, loss in val_cos.items()}
        
        return avg_val_loss, avg_val_losses, avg_val_mse, avg_val_cos
    
    def save_model(self, epoch):
        save_path = os.path.join(self.output_dir, f"model_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'eeg_model_state_dict': self.eeg_model.state_dict(),
            'image_diffusion_state_dict': self.diffusion_models['image'].diffusion_prior.state_dict(),
            'text_diffusion_state_dict': self.diffusion_models['text'].diffusion_prior.state_dict(),
            'superpixel_few_diffusion_state_dict': self.diffusion_models['superpixel_few'].diffusion_prior.state_dict(),
            'superpixel_middle_diffusion_state_dict': self.diffusion_models['superpixel_middle'].diffusion_prior.state_dict(),
            'superpixel_large_diffusion_state_dict': self.diffusion_models['superpixel_large'].diffusion_prior.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_path)
        print(f"Model saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Multimodal EEG Diffusion Training')
    parser.add_argument('--data_path', type=str, default="../../Data/Preprocessed_data_250Hz", 
                        help='Path to the EEG dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs/multimodal_diffusion', 
                        help='Directory to save output results')
    parser.add_argument('--project', type=str, default="multimodal_eeg_diffusion", 
                        help='WandB project name')
    parser.add_argument('--entity', type=str, default="my_entity", 
                        help='WandB entity name')
    parser.add_argument('--name', type=str, default="multimodal_baseline", 
                        help='Experiment name')
    parser.add_argument('--lr_eeg', type=float, default=3e-4, help='Learning rate for EEG model')
    parser.add_argument('--lr_diff', type=float, default=1e-4, help='Learning rate for diffusion models')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--gpu', type=str, default='cuda', help='GPU device to use')
    parser.add_argument('--subjects', nargs='+', default=['sub-01'], 
                        help='List of subject IDs')
    parser.add_argument('--superpixel_base_path', type=str, 
                        default='../../Data/processed_images',
                        help='Base path to superpixel features')
    
    args = parser.parse_args()
    
    trainer = MultimodalDiffusionTrainer(args)
    trainer.train()

if __name__ == '__main__':
    main()