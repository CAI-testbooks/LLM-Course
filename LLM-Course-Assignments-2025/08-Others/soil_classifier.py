"""
土壤类型分类微调 - 修复版本
自动检测实际土壤类型数量
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from config import config
from eurosat_loader import EuroSATFeatureExtractor, Config as EuroSATConfig

class SoilDataset(Dataset):
    """土壤类型数据集"""

    def __init__(self, features: np.ndarray, labels: np.ndarray,
                 soil_types: List[str], transform=None, augment: bool = False):
        """
        初始化数据集

        Args:
            features: 特征数组 [n_samples, feature_dim]
            labels: 标签数组 [n_samples]
            soil_types: 实际土壤类型列表
            transform: 数据转换
            augment: 是否数据增强
        """
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.soil_types = soil_types
        self.num_classes = len(soil_types)
        self.transform = transform
        self.augment = augment

        # 验证标签范围
        unique_labels = np.unique(self.labels)
        max_label = self.labels.max()

        if max_label >= self.num_classes:
            print(f"警告: 标签值 {max_label} 超出类别范围 {self.num_classes-1}")
            # 重新映射标签
            label_mapping = {old: new for new, old in enumerate(unique_labels)}
            self.labels = np.array([label_mapping[l] for l in self.labels])

        # 统计
        self.class_counts = np.bincount(self.labels, minlength=self.num_classes)

        print(f"数据集: {len(self.features)} 样本, {self.num_classes} 类别")
        print(f"类别分布: {dict(zip(self.soil_types, self.class_counts))}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        # 数据增强
        if self.augment and self.transform:
            feature = self.transform(feature)

        # 转换为张量
        feature = torch.from_numpy(feature)

        return {
            'features': feature,
            'labels': label,
            'idx': idx
        }

    def get_class_weights(self):
        """获取类别权重（用于处理不平衡）"""
        total = len(self.labels)
        class_counts = self.class_counts
        # 避免除以零
        class_counts = np.where(class_counts == 0, 1, class_counts)
        weights = total / (self.num_classes * class_counts)
        return torch.FloatTensor(weights)


class SoilClassifier(nn.Module):
    """土壤类型分类器"""

    def __init__(self, feature_dim: int = 512, num_classes: int = None):
        super(SoilClassifier, self).__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # 特征投影层
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        # 分类头
        if num_classes is not None:
            self.classifier = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(64, num_classes)
            )
        else:
            # 延迟创建分类头
            self.classifier = None

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def build_classifier(self, num_classes: int):
        """动态构建分类头"""
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        ).to(next(self.parameters()).device)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        前向传播

        Args:
            x: 输入特征 [batch_size, feature_dim]
            return_attention: 是否返回注意力权重

        Returns:
            分类logits或(logits, attention_weights)
        """
        if self.classifier is None:
            raise ValueError("请先调用 build_classifier 设置类别数")

        # 特征投影
        projected = self.projection(x)

        # 注意力权重
        attention_weights = torch.softmax(self.attention(projected), dim=0)
        weighted_features = projected * attention_weights

        # 分类
        logits = self.classifier(weighted_features)

        if return_attention:
            return logits, attention_weights.squeeze()
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征（用于迁移学习）"""
        return self.projection(x)


class SoilDataLoader:
    """土壤数据加载器 - 修复版本"""

    def __init__(self, config_path: str = None):
        self.config = config
        self.device = self.config.get_device()

        # 自动检测土壤类型
        self.actual_soil_types = None
        self.soil_mapping = {}

    def detect_soil_types(self, eurosat_labels: np.ndarray) -> Tuple[List[str], np.ndarray]:
        """检测实际存在的土壤类型并重新映射标签"""

        soil_config = self.config.soil_config
        eurosat_to_soil = soil_config["eurosat_to_soil"]

        # 收集所有出现的土壤类型
        unique_soil_indices = set()

        for label in eurosat_labels:
            soil_idx = eurosat_to_soil.get(int(label), -1)
            if soil_idx != -1:  # 有效的土壤类型
                unique_soil_indices.add(soil_idx)

        # 排序并创建映射
        sorted_soil_indices = sorted(unique_soil_indices)

        # 创建新的映射：原土壤索引 -> 新连续索引
        new_mapping = {old: new for new, old in enumerate(sorted_soil_indices)}

        # 获取对应的土壤类型名称
        soil_types = [soil_config["soil_types"][idx] for idx in sorted_soil_indices]

        print(f"检测到 {len(soil_types)} 种土壤类型: {soil_types}")
        print(f"土壤类型映射: {new_mapping}")

        return soil_types, new_mapping

    def load_eurosat_features(self) -> Dict[str, np.ndarray]:
        """加载EuroSAT特征并自动检测土壤类型"""
        print("加载EuroSAT特征...")

        features_dir = Path(self.config.paths["eurosat_features"])

        datasets = {}

        # 先加载训练集以确定土壤类型
        train_file = features_dir / "train_features.pkl"

        if not train_file.exists():
            raise FileNotFoundError(f"训练特征文件不存在: {train_file}")

        with open(train_file, 'rb') as f:
            train_data = pickle.load(f)

        # 检测土壤类型
        self.actual_soil_types, self.soil_mapping = self.detect_soil_types(train_data['labels'])

        for split in ['train', 'val', 'test']:
            feature_file = features_dir / f"{split}_features.pkl"

            if not feature_file.exists():
                print(f"警告: {split}特征文件不存在: {feature_file}")
                continue

            with open(feature_file, 'rb') as f:
                data = pickle.load(f)

            features = data['features']
            eurosat_labels = data['labels']

            # 将EuroSAT标签转换为土壤类型标签并进行重新映射
            soil_labels = []
            valid_indices = []

            for i, label in enumerate(eurosat_labels):
                original_soil_idx = self.config.soil_config["eurosat_to_soil"].get(int(label), -1)
                if original_soil_idx != -1:  # 有效的土壤类型
                    # 重新映射到连续索引
                    new_soil_idx = self.soil_mapping.get(original_soil_idx)
                    if new_soil_idx is not None:
                        soil_labels.append(new_soil_idx)
                        valid_indices.append(i)

            # 过滤无效样本
            if valid_indices:
                features = features[valid_indices]
                soil_labels = np.array(soil_labels)
            else:
                print(f"警告: {split}集没有有效的土壤类型样本")
                continue

            datasets[split] = {
                'features': features,
                'labels': soil_labels,
                'original_labels': eurosat_labels[valid_indices] if valid_indices else [],
                'soil_types': self.actual_soil_types
            }

            print(f"{split}: {len(features)} 样本, {len(np.unique(soil_labels))} 土壤类型")

        return datasets

    def create_datasets(self) -> Dict[str, SoilDataset]:
        """创建数据集"""
        data_dict = self.load_eurosat_features()

        datasets = {}
        for split in ['train', 'val', 'test']:
            if split in data_dict:
                data = data_dict[split]
                datasets[split] = SoilDataset(
                    features=data['features'],
                    labels=data['labels'],
                    soil_types=self.actual_soil_types,
                    augment=(split == 'train')
                )

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


class SoilFinetuner:
    """土壤类型微调器 - 修复版本"""

    def __init__(self, config_path: str = None):
        self.config = config
        self.device = self.config.get_device()

        # 数据加载器
        self.data_loader = SoilDataLoader()

        # 模型
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None

        # 训练历史
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': []
        }

        # 输出目录
        self.output_dir = Path(self.config.paths["finetune_output"]) / "soil_classifier"
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

    def build_model(self, pretrained_path: str = None, num_classes: int = None) -> nn.Module:
        """构建模型"""
        print("构建土壤分类模型...")

        # 如果未指定类别数，使用配置中的默认值
        if num_classes is None:
            num_classes = self.config.soil_config["num_classes"]

        # 加载EuroSAT预训练模型
        eurosat_config = EuroSATConfig()
        base_model = EuroSATFeatureExtractor(eurosat_config)

        if pretrained_path and Path(pretrained_path).exists():
            checkpoint = torch.load(pretrained_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                base_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                base_model.load_state_dict(checkpoint)
            print(f"加载预训练权重: {pretrained_path}")
        else:
            print("警告: 未找到预训练权重，使用随机初始化")

        # 冻结卷积层
        for param in base_model.conv_layers.parameters():
            param.requires_grad = False

        # 创建土壤分类器（先不指定类别数）
        self.model = SoilClassifier(feature_dim=512)

        # 稍后动态构建分类头
        self.model.build_classifier(num_classes)

        # 将模型移动到设备
        self.model = self.model.to(self.device)

        # 损失函数（带类别权重）
        self.criterion = nn.CrossEntropyLoss()

        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.training_config["learning_rate"],
            weight_decay=self.config.training_config["weight_decay"]
        )

        # 学习率调度器
        self.scheduler = self._create_scheduler()

        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters())}")
        print(f"可训练参数量: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

        return self.model

    def _create_scheduler(self):
        """创建学习率调度器"""
        scheduler_config = self.config.training_config["scheduler"]

        if scheduler_config["name"] == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training_config["epochs"],
                eta_min=scheduler_config["min_lr"]
            )
        elif scheduler_config["name"] == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=5,
                factor=0.5,
                min_lr=scheduler_config["min_lr"]
            )
        else:
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=20,
                gamma=0.1
            )

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        from tqdm import tqdm
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} [训练]')

        for batch_idx, batch in enumerate(pbar):
            features = batch['features'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)

            # 前向传播
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(features)
            loss = self.criterion(logits, labels)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training_config["gradient_clip"]
            )

            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def validate(self, dataloader: DataLoader, epoch: int) -> Tuple[float, float]:
        """验证"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            from tqdm import tqdm
            pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} [验证]')

            for batch_idx, batch in enumerate(pbar):
                features = batch['features'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)

                # 前向传播
                logits = self.model(features)
                loss = self.criterion(logits, labels)

                # 统计
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # 收集预测结果
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100. * correct / total:.2f}%'
                })

        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy, all_preds, all_labels

    def test(self, dataloader: DataLoader, dataset: SoilDataset) -> Dict:
        """测试"""
        self.model.eval()

        all_preds = []
        all_labels = []
        all_features = []

        with torch.no_grad():
            from tqdm import tqdm
            for batch in tqdm(dataloader, desc="测试"):
                features = batch['features'].to(self.device)
                labels = batch['labels'].numpy()

                # 前向传播
                logits = self.model(features)
                _, predicted = torch.max(logits, 1)

                # 提取特征
                features_extracted = self.model.extract_features(features)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels)
                all_features.extend(features_extracted.cpu().numpy())

        # 计算指标
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

        accuracy = accuracy_score(all_labels, all_preds)

        # 使用数据集的实际土壤类型名称
        soil_types = dataset.soil_types if hasattr(dataset, 'soil_types') else self.config.soil_config["soil_types"]

        # 确保标签在类别范围内
        max_label = max(max(all_labels), max(all_preds)) if all_labels and all_preds else 0
        if max_label >= len(soil_types):
            print(f"警告: 标签值 {max_label} 超出类别范围 {len(soil_types)-1}")
            # 截断土壤类型列表
            soil_types = soil_types[:max_label+1]

        report = classification_report(
            all_labels, all_preds,
            target_names=soil_types,
            output_dict=True
        )
        cm = confusion_matrix(all_labels, all_preds)

        return {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'labels': all_labels,
            'features': np.array(all_features),
            'soil_types': soil_types
        }

    def train(self, dataloaders: Dict[str, DataLoader]):
        """训练模型"""
        print("开始训练土壤类型分类模型...")

        # 获取实际类别数
        if 'train' in dataloaders and hasattr(dataloaders['train'].dataset, 'num_classes'):
            actual_num_classes = dataloaders['train'].dataset.num_classes
            print(f"检测到实际类别数: {actual_num_classes}")
        else:
            # 从数据集中推断
            for batch in dataloaders['train']:
                labels = batch['labels']
                actual_num_classes = len(torch.unique(labels))
                break

        # 重新构建模型（如果需要）
        if self.model is None or self.model.num_classes != actual_num_classes:
            print(f"重新构建模型，类别数: {actual_num_classes}")
            self.build_model(self.config.paths["eurosat_model"], actual_num_classes)

        best_val_acc = 0
        patience_counter = 0
        best_model_state = None

        epochs = self.config.training_config["epochs"]
        patience = self.config.training_config["patience"]

        for epoch in range(epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(dataloaders['train'], epoch)

            # 验证
            val_loss, val_acc, val_preds, val_labels = self.validate(dataloaders['val'], epoch)

            # 更新学习率
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # 记录历史
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)

            self.logger.info(
                f"Epoch {epoch+1}/{epochs}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                f"LR: {current_lr:.6f}"
            )

            # 早停和保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()

                # 保存最佳模型
                self.save_checkpoint(epoch, 'best')
                self.logger.info(f"最佳模型保存，验证准确率: {val_acc:.2f}%")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.logger.info(f"早停触发，在epoch {epoch+1}")
                    break

        # 加载最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        # 保存最终模型
        self.save_checkpoint(epochs - 1, 'final')

        # 保存训练历史
        self.save_history()

        self.logger.info(f"训练完成，最佳验证准确率: {best_val_acc:.2f}%")

        return self.history

    def save_checkpoint(self, epoch: int, name: str):
        """保存检查点"""
        checkpoint_path = self.output_dir / f"{name}_checkpoint.pth"

        # 获取土壤类型信息
        soil_types = []
        if hasattr(self, 'data_loader') and hasattr(self.data_loader, 'actual_soil_types'):
            soil_types = self.data_loader.actual_soil_types

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'history': self.history,
            'config': self.config,
            'soil_types': soil_types,
            'num_classes': self.model.num_classes if self.model else None
        }, checkpoint_path)

        self.logger.info(f"检查点保存到: {checkpoint_path}")

    def save_history(self):
        """保存训练历史"""
        history_path = self.output_dir / "training_history.json"

        import json
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        # 保存为numpy格式
        np.save(self.output_dir / "history.npy", self.history)

    def visualize_results(self, test_results: Dict):
        """可视化结果"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            viz_dir = self.output_dir / "visualizations"
            viz_dir.mkdir(parents=True, exist_ok=True)

            # 1. 训练历史
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # 损失曲线
            axes[0, 0].plot(self.history['train_loss'], label='训练损失')
            axes[0, 0].plot(self.history['val_loss'], label='验证损失')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('损失')
            axes[0, 0].set_title('训练和验证损失')
            axes[0, 0].legend()
            axes[0, 0].grid(True)

            # 准确率曲线
            axes[0, 1].plot(self.history['train_acc'], label='训练准确率')
            axes[0, 1].plot(self.history['val_acc'], label='验证准确率')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('准确率 (%)')
            axes[0, 1].set_title('训练和验证准确率')
            axes[0, 1].legend()
            axes[0, 1].grid(True)

            # 学习率曲线
            axes[1, 0].plot(self.history['lr'])
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('学习率')
            axes[1, 0].set_title('学习率变化')
            axes[1, 0].grid(True)

            # 混淆矩阵
            cm = test_results['confusion_matrix']
            soil_types = test_results.get('soil_types', ['Class 0', 'Class 1', 'Class 2', 'Class 3'][:len(cm)])

            im = axes[1, 1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            axes[1, 1].set_title('混淆矩阵')
            plt.colorbar(im, ax=axes[1, 1])

            # 设置刻度
            tick_marks = np.arange(len(soil_types))
            axes[1, 1].set_xticks(tick_marks)
            axes[1, 1].set_xticklabels(soil_types, rotation=45)
            axes[1, 1].set_yticks(tick_marks)
            axes[1, 1].set_yticklabels(soil_types)

            # 添加数值
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    axes[1, 1].text(j, i, format(cm[i, j], 'd'),
                                  horizontalalignment="center",
                                  color="white" if cm[i, j] > thresh else "black")

            plt.tight_layout()
            plt.savefig(viz_dir / "training_results.png", dpi=300, bbox_inches='tight')
            plt.close()

            # 2. 特征可视化（t-SNE）
            try:
                from sklearn.manifold import TSNE

                features = test_results['features']
                labels = test_results['labels']

                if len(features) > 50:
                    # 采样
                    n_samples = min(500, len(features))
                    indices = np.random.choice(len(features), n_samples, replace=False)
                    features_sample = features[indices]
                    labels_sample = [labels[i] for i in indices]

                    # t-SNE
                    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
                    features_2d = tsne.fit_transform(features_sample)

                    # 绘制
                    plt.figure(figsize=(10, 8))
                    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                                        c=labels_sample, cmap='tab10', alpha=0.7, s=50)

                    plt.colorbar(scatter, label='土壤类型')
                    plt.xlabel('t-SNE 维度 1')
                    plt.ylabel('t-SNE 维度 2')
                    plt.title('土壤类型特征空间可视化 (t-SNE)')
                    plt.tight_layout()
                    plt.savefig(viz_dir / "feature_tsne.png", dpi=300, bbox_inches='tight')
                    plt.close()
            except Exception as e:
                self.logger.warning(f"t-SNE可视化失败: {e}")

            self.logger.info(f"可视化结果保存到: {viz_dir}")

        except ImportError as e:
            self.logger.warning(f"可视化依赖库未安装: {e}")

    def run_pipeline(self):
        """运行完整微调流水线"""
        self.logger.info("=" * 60)
        self.logger.info("🌱 土壤类型微调流水线")
        self.logger.info("=" * 60)

        try:
            # 1. 准备数据
            self.logger.info("步骤1: 准备数据...")
            dataloaders = self.data_loader.create_dataloaders()

            if 'train' not in dataloaders or 'val' not in dataloaders:
                raise ValueError("缺少训练集或验证集")

            # 2. 构建模型
            self.logger.info("步骤2: 构建模型...")
            self.build_model(self.config.paths["eurosat_model"])

            # 3. 训练模型
            self.logger.info("步骤3: 训练模型...")
            history = self.train(dataloaders)

            # 4. 测试模型
            self.logger.info("步骤4: 测试模型...")
            if 'test' in dataloaders:
                test_dataset = dataloaders['test'].dataset
                test_results = self.test(dataloaders['test'], test_dataset)

                self.logger.info(f"测试准确率: {test_results['accuracy']:.4f}")

                # 保存测试结果
                test_path = self.output_dir / "test_results.pkl"
                with open(test_path, 'wb') as f:
                    import pickle
                    pickle.dump(test_results, f)

                self.logger.info(f"测试结果保存到: {test_path}")

                # 5. 可视化
                self.logger.info("步骤5: 可视化...")
                self.visualize_results(test_results)

            self.logger.info("=" * 60)
            self.logger.info("✅ 土壤类型微调完成!")
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
    finetuner = SoilFinetuner()

    # 运行微调流水线
    results = finetuner.run_pipeline()

    return results


if __name__ == "__main__":
    main()