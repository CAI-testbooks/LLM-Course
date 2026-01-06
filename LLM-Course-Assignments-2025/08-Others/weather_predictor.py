"""
气象预测模型微调 - 修复维度版本
自动检测输入特征维度
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

from config import config


class WeatherDataset(Dataset):
    """气象时序数据集"""

    def __init__(self, features: np.ndarray, targets: np.ndarray,
                 window_size: int = 7, forecast_horizon: int = 1):
        """
        初始化数据集

        Args:
            features: 特征数组 [n_samples, n_timesteps, n_features]
            targets: 目标数组 [n_samples, 2] (Next_Tmax, Next_Tmin)
            window_size: 时间窗口大小
            forecast_horizon: 预测步长
        """
        self.features = features.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.feature_dim = features.shape[-1] if len(features.shape) > 2 else features.shape[1]

        # 验证数据形状
        assert len(self.features) == len(self.targets), "特征和目标长度不一致"

        print(f"气象数据集: {len(self.features)} 样本")
        print(f"特征形状: {self.features.shape}")
        print(f"目标形状: {self.targets.shape}")
        print(f"特征维度: {self.feature_dim}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # 获取窗口特征
        window_features = self.features[idx]

        # 获取目标
        window_targets = self.targets[idx]

        # 转换为张量
        features_tensor = torch.from_numpy(window_features)
        targets_tensor = torch.from_numpy(window_targets)

        return {
            'features': features_tensor,
            'targets': targets_tensor,
            'idx': idx,
            'feature_dim': self.feature_dim
        }


class WeatherPredictor(nn.Module):
    """气象预测模型（LSTM + Attention）"""

    def __init__(self, input_size: int = None, hidden_size: int = 128,
                 num_layers: int = 2, output_size: int = 2, dropout: float = 0.2):
        super(WeatherPredictor, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # 如果未指定input_size，延迟创建LSTM
        self.lstm = None
        if input_size is not None:
            self._build_lstm(input_size)

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

        # 层归一化
        self.ln = nn.LayerNorm(hidden_size * 2)

    def _build_lstm(self, input_size: int):
        """构建LSTM层"""
        self.input_size = input_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.2 if self.num_layers > 1 else 0,
            bidirectional=True
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        前向传播

        Args:
            x: 输入序列 [batch_size, seq_len, input_size]
            return_attention: 是否返回注意力权重

        Returns:
            预测结果或(预测结果, 注意力权重)
        """
        if self.lstm is None:
            # 动态构建LSTM
            input_size = x.size(-1)
            self._build_lstm(input_size)
            self.lstm = self.lstm.to(x.device)

        batch_size = x.size(0)

        # 验证输入维度
        if x.size(-1) != self.input_size:
            raise ValueError(f"输入维度不匹配: 期望 {self.input_size}, 实际 {x.size(-1)}")

        # LSTM编码
        lstm_out, (hidden, cell) = self.lstm(x)  # [batch, seq_len, hidden*2]
        lstm_out = self.ln(lstm_out)

        # 注意力权重
        attention_weights = torch.softmax(
            self.attention(lstm_out).squeeze(-1), dim=1
        ).unsqueeze(2)  # [batch, seq_len, 1]

        # 上下文向量
        context = torch.sum(lstm_out * attention_weights, dim=1)  # [batch, hidden*2]

        # 预测
        predictions = self.fc(context)  # [batch, output_size]

        if return_attention:
            return predictions, attention_weights.squeeze()
        return predictions

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征"""
        if self.lstm is None:
            input_size = x.size(-1)
            self._build_lstm(input_size)
            self.lstm = self.lstm.to(x.device)

        lstm_out, _ = self.lstm(x)
        lstm_out = self.ln(lstm_out)

        # 注意力权重
        attention_weights = torch.softmax(
            self.attention(lstm_out).squeeze(-1), dim=1
        ).unsqueeze(2)

        # 上下文向量
        context = torch.sum(lstm_out * attention_weights, dim=1)

        return context


class WeatherDataLoader:
    """气象数据加载器 - 修复版本"""

    def __init__(self):
        self.config = config
        self.feature_dim = None

    def load_weather_samples(self) -> Dict[str, np.ndarray]:
        """加载气象时序样本"""
        print("加载气象时序样本...")

        samples_path = Path(self.config.paths["weather_ts_samples"])

        if not samples_path.exists():
            raise FileNotFoundError(f"气象样本文件不存在: {samples_path}")

        with open(samples_path, 'rb') as f:
            samples = pickle.load(f)

        # 提取特征和目标
        features = []
        targets = []

        for sample in samples:
            # 特征: [window_size, feature_dim]
            window_features = sample['features']
            features.append(window_features)

            # 目标: [2] (Next_Tmax, Next_Tmin)
            target_tmax = sample['targets']['Next_Tmax']
            target_tmin = sample['targets']['Next_Tmin']
            targets.append([target_tmax, target_tmin])

        features = np.array(features)
        targets = np.array(targets)

        # 获取实际特征维度
        if len(features.shape) == 3:
            self.feature_dim = features.shape[-1]
        else:
            self.feature_dim = features.shape[1] if len(features.shape) > 1 else 1

        print(f"加载 {len(features)} 个气象样本")
        print(f"特征形状: {features.shape}")
        print(f"目标形状: {targets.shape}")
        print(f"实际特征维度: {self.feature_dim}")

        # 更新配置中的特征维度
        if self.feature_dim != self.config.weather_config["input_features"]:
            print(f"更新配置特征维度: {self.config.weather_config['input_features']} -> {self.feature_dim}")
            self.config.weather_config["input_features"] = self.feature_dim

        return {
            'features': features,
            'targets': targets,
            'feature_dim': self.feature_dim
        }

    def create_datasets(self, split_ratio: Tuple = (0.7, 0.15, 0.15)) -> Dict[str, WeatherDataset]:
        """创建数据集"""
        data = self.load_weather_samples()

        n_samples = len(data['features'])
        indices = np.random.permutation(n_samples)

        train_size = int(n_samples * split_ratio[0])
        val_size = int(n_samples * split_ratio[1])

        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]

        datasets = {}

        for name, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
            datasets[name] = WeatherDataset(
                features=data['features'][idx],
                targets=data['targets'][idx],
                window_size=self.config.weather_config["window_size"]
            )

        print(f"数据集划分: 训练集={len(datasets['train'])}, "
              f"验证集={len(datasets['val'])}, 测试集={len(datasets['test'])}")

        return datasets

    def create_dataloaders(self, batch_size: int = None) -> Dict[str, DataLoader]:
        """创建数据加载器"""
        if batch_size is None:
            batch_size = self.config.training_config["batch_size"]

        datasets = self.create_datasets()
        dataloaders = {}

        for split in ['train', 'val', 'test']:
            if split in datasets:
                shuffle = (split == 'train')
                dataloaders[split] = DataLoader(
                    datasets[split],
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=4,
                    pin_memory=True
                )

        return dataloaders


class WeatherFinetuner:
    """气象预测微调器 - 修复版本"""

    def __init__(self):
        self.config = config
        self.device = self.config.get_device()

        # 数据加载器
        self.data_loader = WeatherDataLoader()

        # 模型
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None

        # 训练历史
        self.history = {
            'train_loss': [], 'train_mae': [], 'train_rmse': [],
            'val_loss': [], 'val_mae': [], 'val_rmse': [],
            'lr': []
        }

        # 输出目录
        self.output_dir = Path(self.config.paths["finetune_output"]) / "weather_predictor"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 日志
        self._setup_logging()

        print(self.config)

    def _setup_logging(self):
        """设置日志"""
        import logging

        log_file = self.output_dir / "training.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger(__name__)

    def analyze_features(self, dataloader: DataLoader):
        """分析特征维度"""
        print("分析特征维度...")

        for batch in dataloader:
            features = batch['features']
            print(f"批次特征形状: {features.shape}")

            # 获取实际维度
            actual_feature_dim = features.shape[-1]
            config_feature_dim = self.config.weather_config["input_features"]

            if actual_feature_dim != config_feature_dim:
                print(f"特征维度不匹配: 配置={config_feature_dim}, 实际={actual_feature_dim}")
                print(f"更新配置...")

                # 更新配置
                self.config.weather_config["input_features"] = actual_feature_dim

                # 保存更新后的配置
                config.save()

            break

        return self.config.weather_config["input_features"]

    def build_model(self, input_size: int = None) -> nn.Module:
        """构建模型"""
        print("构建气象预测模型...")

        weather_config = self.config.weather_config

        # 使用实际特征维度
        if input_size is None:
            input_size = weather_config["input_features"]

        print(f"使用输入特征维度: {input_size}")

        self.model = WeatherPredictor(
            input_size=input_size,
            hidden_size=weather_config["hidden_size"],
            num_layers=weather_config["num_layers"],
            output_size=weather_config["output_features"],
            dropout=0.2
        )

        # 移动到设备
        self.model = self.model.to(self.device)

        # 损失函数 (Huber损失对异常值更鲁棒)
        self.criterion = nn.HuberLoss(delta=1.0)

        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.training_config["learning_rate"],
            weight_decay=self.config.training_config["weight_decay"]
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )

        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters())}")

        return self.model

    def calculate_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict:
        """计算评估指标"""
        predictions_np = predictions.cpu().detach().numpy()
        targets_np = targets.cpu().detach().numpy()

        # MAE
        mae = np.mean(np.abs(predictions_np - targets_np))

        # RMSE
        rmse = np.sqrt(np.mean((predictions_np - targets_np) ** 2))

        # R² Score
        ss_res = np.sum((targets_np - predictions_np) ** 2)
        ss_tot = np.sum((targets_np - np.mean(targets_np)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2)
        }

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Tuple[float, Dict]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_mae = 0
        total_rmse = 0

        from tqdm import tqdm
        pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1} [训练]')

        for batch_idx, batch in enumerate(pbar):
            features = batch['features'].to(self.device, non_blocking=True)
            targets = batch['targets'].to(self.device, non_blocking=True)

            # 验证特征维度
            expected_dim = self.config.weather_config["input_features"]
            actual_dim = features.shape[-1]

            if actual_dim != expected_dim:
                print(f"警告: 特征维度不匹配! 批次 {batch_idx}: 期望 {expected_dim}, 实际 {actual_dim}")
                # 调整模型输入维度
                if hasattr(self.model, '_build_lstm'):
                    self.model._build_lstm(actual_dim)
                    self.model.lstm = self.model.lstm.to(self.device)

            # 前向传播
            self.optimizer.zero_grad(set_to_none=True)
            predictions = self.model(features)
            loss = self.criterion(predictions, targets)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training_config["gradient_clip"]
            )

            self.optimizer.step()

            # 计算指标
            metrics = self.calculate_metrics(predictions, targets)

            # 统计
            total_loss += loss.item()
            total_mae += metrics['mae']
            total_rmse += metrics['rmse']

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mae': f'{metrics["mae"]:.3f}',
                'rmse': f'{metrics["rmse"]:.3f}'
            })

        avg_loss = total_loss / len(dataloader)
        avg_mae = total_mae / len(dataloader)
        avg_rmse = total_rmse / len(dataloader)

        return avg_loss, {'mae': avg_mae, 'rmse': avg_rmse}

    def validate(self, dataloader: DataLoader, epoch: int) -> Tuple[float, Dict]:
        """验证"""
        self.model.eval()
        total_loss = 0
        total_mae = 0
        total_rmse = 0

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            from tqdm import tqdm
            pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1} [验证]')

            for batch_idx, batch in enumerate(pbar):
                features = batch['features'].to(self.device, non_blocking=True)
                targets = batch['targets'].to(self.device, non_blocking=True)

                # 验证特征维度
                expected_dim = self.config.weather_config["input_features"]
                actual_dim = features.shape[-1]

                if actual_dim != expected_dim:
                    print(f"警告: 验证集特征维度不匹配! 批次 {batch_idx}: 期望 {expected_dim}, 实际 {actual_dim}")

                # 前向传播
                predictions = self.model(features)
                loss = self.criterion(predictions, targets)

                # 计算指标
                metrics = self.calculate_metrics(predictions, targets)

                # 统计
                total_loss += loss.item()
                total_mae += metrics['mae']
                total_rmse += metrics['rmse']

                # 收集结果
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'mae': f'{metrics["mae"]:.3f}',
                    'rmse': f'{metrics["rmse"]:.3f}'
                })

        avg_loss = total_loss / len(dataloader)
        avg_mae = total_mae / len(dataloader)
        avg_rmse = total_rmse / len(dataloader)

        return avg_loss, {'mae': avg_mae, 'rmse': avg_rmse}, all_predictions, all_targets

    def test(self, dataloader: DataLoader) -> Dict:
        """测试"""
        self.model.eval()

        all_predictions = []
        all_targets = []
        all_features = []

        with torch.no_grad():
            from tqdm import tqdm
            for batch in tqdm(dataloader, desc="测试"):
                features = batch['features'].to(self.device)
                targets = batch['targets'].numpy()

                # 前向传播
                predictions = self.model(features)

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets)
                all_features.extend(features.cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        # 计算详细指标
        metrics = {}

        # 整体指标
        metrics['mae'] = np.mean(np.abs(all_predictions - all_targets))
        metrics['rmse'] = np.sqrt(np.mean((all_predictions - all_targets) ** 2))

        # 分项指标 (Tmax, Tmin)
        for i, name in enumerate(['Tmax', 'Tmin']):
            pred = all_predictions[:, i]
            true = all_targets[:, i]

            mae = np.mean(np.abs(pred - true))
            rmse = np.sqrt(np.mean((pred - true) ** 2))
            r2 = 1 - np.sum((true - pred) ** 2) / np.sum((true - np.mean(true)) ** 2)

            metrics[f'{name}_mae'] = float(mae)
            metrics[f'{name}_rmse'] = float(rmse)
            metrics[f'{name}_r2'] = float(r2)

        return {
            'metrics': metrics,
            'predictions': all_predictions,
            'targets': all_targets,
            'features': np.array(all_features)
        }

    def train(self, dataloaders: Dict[str, DataLoader]):
        """训练模型"""
        print("开始训练气象预测模型...")

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        epochs = self.config.training_config["epochs"]
        patience = self.config.training_config["patience"]

        for epoch in range(epochs):
            # 训练
            train_loss, train_metrics = self.train_epoch(dataloaders['train'], epoch)

            # 验证
            val_loss, val_metrics, val_preds, val_targets = self.validate(dataloaders['val'], epoch)

            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step()

            # 记录历史
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['train_mae'].append(train_metrics['mae'])
            self.history['train_rmse'].append(train_metrics['rmse'])
            self.history['val_loss'].append(val_loss)
            self.history['val_mae'].append(val_metrics['mae'])
            self.history['val_rmse'].append(val_metrics['rmse'])
            self.history['lr'].append(current_lr)

            self.logger.info(
                f"Epoch {epoch + 1}/{epochs}: "
                f"Train Loss: {train_loss:.4f}, MAE: {train_metrics['mae']:.3f}, RMSE: {train_metrics['rmse']:.3f} | "
                f"Val Loss: {val_loss:.4f}, MAE: {val_metrics['mae']:.3f}, RMSE: {val_metrics['rmse']:.3f} | "
                f"LR: {current_lr:.6f}"
            )

            # 早停和保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()

                # 保存最佳模型
                self.save_checkpoint(epoch, 'best')
                self.logger.info(f"最佳模型保存，验证损失: {val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.logger.info(f"早停触发，在epoch {epoch + 1}")
                    break

        # 加载最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        # 保存最终模型
        self.save_checkpoint(epochs - 1, 'final')

        # 保存训练历史
        self.save_history()

        self.logger.info(f"训练完成，最佳验证损失: {best_val_loss:.4f}")

        return self.history

    def save_checkpoint(self, epoch: int, name: str):
        """保存检查点"""
        checkpoint_path = self.output_dir / f"{name}_checkpoint.pth"

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'history': self.history,
            'config': self.config,
            'input_size': self.config.weather_config["input_features"]
        }, checkpoint_path)

        self.logger.info(f"检查点保存到: {checkpoint_path}")

    def save_history(self):
        """保存训练历史"""
        history_path = self.output_dir / "training_history.json"

        import json
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        np.save(self.output_dir / "history.npy", self.history)

    def visualize_results(self, test_results: Dict):
        """可视化结果"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            viz_dir = self.output_dir / "visualizations"
            viz_dir.mkdir(parents=True, exist_ok=True)

            # 1. 训练历史
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))

            # 损失曲线
            axes[0, 0].plot(self.history['train_loss'], label='训练损失')
            axes[0, 0].plot(self.history['val_loss'], label='验证损失')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('损失')
            axes[0, 0].set_title('训练和验证损失')
            axes[0, 0].legend()
            axes[0, 0].grid(True)

            # MAE曲线
            axes[0, 1].plot(self.history['train_mae'], label='训练MAE')
            axes[0, 1].plot(self.history['val_mae'], label='验证MAE')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('MAE (°C)')
            axes[0, 1].set_title('平均绝对误差')
            axes[0, 1].legend()
            axes[0, 1].grid(True)

            # RMSE曲线
            axes[0, 2].plot(self.history['train_rmse'], label='训练RMSE')
            axes[0, 2].plot(self.history['val_rmse'], label='验证RMSE')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('RMSE (°C)')
            axes[0, 2].set_title('均方根误差')
            axes[0, 2].legend()
            axes[0, 2].grid(True)

            # 学习率曲线
            axes[1, 0].plot(self.history['lr'])
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('学习率')
            axes[1, 0].set_title('学习率变化')
            axes[1, 0].grid(True)

            # 预测 vs 真实值 (Tmax)
            predictions = test_results['predictions']
            targets = test_results['targets']

            # Tmax散点图
            axes[1, 1].scatter(targets[:, 0], predictions[:, 0], alpha=0.5, s=10)
            axes[1, 1].plot([targets[:, 0].min(), targets[:, 0].max()],
                            [targets[:, 0].min(), targets[:, 0].max()], 'r--')
            axes[1, 1].set_xlabel('真实 Tmax (°C)')
            axes[1, 1].set_ylabel('预测 Tmax (°C)')
            axes[1, 1].set_title('最高温度预测')
            axes[1, 1].grid(True)

            # Tmin散点图
            axes[1, 2].scatter(targets[:, 1], predictions[:, 1], alpha=0.5, s=10)
            axes[1, 2].plot([targets[:, 1].min(), targets[:, 1].max()],
                            [targets[:, 1].min(), targets[:, 1].max()], 'r--')
            axes[1, 2].set_xlabel('真实 Tmin (°C)')
            axes[1, 2].set_ylabel('预测 Tmin (°C)')
            axes[1, 2].set_title('最低温度预测')
            axes[1, 2].grid(True)

            plt.tight_layout()
            plt.savefig(viz_dir / "training_results.png", dpi=300, bbox_inches='tight')
            plt.close()

            # 2. 误差分布
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Tmax误差分布
            tmax_errors = predictions[:, 0] - targets[:, 0]
            axes[0].hist(tmax_errors, bins=50, alpha=0.7, edgecolor='black')
            axes[0].axvline(x=0, color='r', linestyle='--')
            axes[0].set_xlabel('预测误差 (°C)')
            axes[0].set_ylabel('频数')
            axes[0].set_title('Tmax预测误差分布')
            axes[0].grid(True, alpha=0.3)

            # Tmin误差分布
            tmin_errors = predictions[:, 1] - targets[:, 1]
            axes[1].hist(tmin_errors, bins=50, alpha=0.7, edgecolor='black', color='orange')
            axes[1].axvline(x=0, color='r', linestyle='--')
            axes[1].set_xlabel('预测误差 (°C)')
            axes[1].set_ylabel('频数')
            axes[1].set_title('Tmin预测误差分布')
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(viz_dir / "error_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"可视化结果保存到: {viz_dir}")

        except ImportError as e:
            self.logger.warning(f"可视化依赖库未安装: {e}")

    def run_pipeline(self):
        """运行完整微调流水线"""
        self.logger.info("=" * 60)
        self.logger.info("🌤️ 气象预测微调流水线")
        self.logger.info("=" * 60)

        try:
            # 1. 准备数据
            self.logger.info("步骤1: 准备数据...")
            dataloaders = self.data_loader.create_dataloaders()

            if 'train' not in dataloaders or 'val' not in dataloaders:
                raise ValueError("缺少训练集或验证集")

            # 2. 分析特征维度
            self.logger.info("步骤2: 分析特征维度...")
            actual_feature_dim = self.analyze_features(dataloaders['train'])

            # 3. 构建模型
            self.logger.info("步骤3: 构建模型...")
            self.build_model(actual_feature_dim)

            # 4. 训练模型
            self.logger.info("步骤4: 训练模型...")
            history = self.train(dataloaders)

            # 5. 测试模型
            self.logger.info("步骤5: 测试模型...")
            if 'test' in dataloaders:
                test_results = self.test(dataloaders['test'])

                metrics = test_results['metrics']
                self.logger.info(f"测试MAE: {metrics['mae']:.3f}°C")
                self.logger.info(f"测试RMSE: {metrics['rmse']:.3f}°C")
                self.logger.info(f"Tmax MAE: {metrics['Tmax_mae']:.3f}°C, R²: {metrics['Tmax_r2']:.3f}")
                self.logger.info(f"Tmin MAE: {metrics['Tmin_mae']:.3f}°C, R²: {metrics['Tmin_r2']:.3f}")

                # 保存测试结果
                test_path = self.output_dir / "test_results.pkl"
                with open(test_path, 'wb') as f:
                    import pickle
                    pickle.dump(test_results, f)

                self.logger.info(f"测试结果保存到: {test_path}")

                # 6. 可视化
                self.logger.info("步骤6: 可视化...")
                self.visualize_results(test_results)

            self.logger.info("=" * 60)
            self.logger.info("✅ 气象预测微调完成!")
            self.logger.info("=" * 60)

            return {
                'model': self.model,
                'history': history,
                'test_results': test_results if 'test' in dataloaders else None
            }

        except Exception as e:
            self.logger.error(f"微调失败: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """主函数"""
    # 创建微调器
    finetuner = WeatherFinetuner()

    # 运行微调流水线
    results = finetuner.run_pipeline()

    return results


if __name__ == "__main__":
    main()