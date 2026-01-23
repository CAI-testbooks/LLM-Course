"""
EuroSAT卫星图像特征提取模块 - 完整版本
用于智能体系统中的卫星图像特征提取和预训练
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import rasterio
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
import pickle
import json
from pathlib import Path


# ============================ 配置参数 ============================
class Config:
    def __init__(self):
        # 数据路径
        self.data_root = "/home/Liyang/agent/EuroSAT_MS"
        self.output_dir = "/home/Liyang/agent/features/eurosat"

        # 类别名称（直接从文件夹获取）
        self.class_names = [
            'AnnualCrop',
            'Forest',
            'HerbaceousVegetat...',
            'Highway',
            'Industrial',
            'Pasture',
            'PermanentCrop',
            'Residential',
            'River',
            'SeaLake'
        ]

        # 数据参数
        self.img_size = 64
        self.batch_size = 32
        self.num_workers = 4

        # 模型参数
        self.num_channels = 13  # EuroSAT多光谱波段数
        self.num_classes = 10  # 10个类别
        self.feature_dim = 512  # 特征向量维度

        # 训练参数
        self.epochs = 30
        self.learning_rate = 0.001
        self.patience = 10  # 早停

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "features"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "visualizations"), exist_ok=True)


# ============================ 数据集类 ============================
class EuroSATDataset(Dataset):
    """EuroSAT数据集加载器"""

    def __init__(self, root_dir, class_names, split='train', transform=None, test_size=0.2, random_seed=42):
        self.root_dir = root_dir
        self.class_names = class_names
        self.split = split
        self.transform = transform

        # 收集所有文件
        self.samples = []
        self.labels = []

        print(f"正在加载{split}数据集...")

        for label_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(root_dir, class_name)

            # 检查目录是否存在
            if not os.path.exists(class_dir):
                print(f"警告: 目录不存在 - {class_dir}")
                continue

            # 获取所有tif文件
            tif_files = [f for f in os.listdir(class_dir) if f.endswith('.tif')]

            if not tif_files:
                print(f"警告: 没有找到tif文件 - {class_dir}")
                continue

            print(f"  类别 {class_name}: {len(tif_files)} 个文件")

            for file_name in tif_files:
                self.samples.append(os.path.join(class_dir, file_name))
                self.labels.append(label_idx)

        # 划分数据集
        if len(self.samples) > 0:
            if split == 'all':
                # 使用所有数据
                pass
            else:
                # 划分训练/验证/测试集
                train_samples, test_samples, train_labels, test_labels = train_test_split(
                    self.samples, self.labels,
                    test_size=test_size,
                    random_state=random_seed,
                    stratify=self.labels
                )

                # 进一步划分验证集
                if split == 'train':
                    self.samples = train_samples
                    self.labels = train_labels
                elif split == 'val':
                    # 从训练集中再分出一部分作为验证集
                    train_samples, val_samples, train_labels, val_labels = train_test_split(
                        train_samples, train_labels,
                        test_size=0.2,  # 训练集的20%作为验证集
                        random_state=random_seed,
                        stratify=train_labels
                    )
                    self.samples = val_samples
                    self.labels = val_labels
                elif split == 'test':
                    self.samples = test_samples
                    self.labels = test_labels

        print(f"  总共: {len(self.samples)} 个样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]

        try:
            # 使用rasterio读取tif文件
            with rasterio.open(img_path) as src:
                # 读取所有波段
                img = src.read()  # 形状: (C, H, W)

                # 转换为float32
                img = img.astype(np.float32)

                # 标准化每个波段到[0,1]
                for i in range(img.shape[0]):
                    band = img[i]
                    min_val = np.min(band)
                    max_val = np.max(band)
                    if max_val > min_val:
                        img[i] = (band - min_val) / (max_val - min_val)

        except Exception as e:
            print(f"读取文件错误: {img_path}, 错误: {e}")
            # 创建随机数据作为占位符
            img = np.random.randn(self.num_channels, 64, 64).astype(np.float32)

        # 转换为PyTorch张量
        img = torch.from_numpy(img)
        label = torch.tensor(label, dtype=torch.long)

        # 应用数据增强
        if self.transform and self.split == 'train':
            img = self.transform(img)

        return {
            'image': img,
            'label': label,
            'path': img_path,
            'class_name': self.class_names[label]
        }

    def get_class_distribution(self):
        """获取类别分布"""
        distribution = {}
        for label in self.labels:
            class_name = self.class_names[label]
            distribution[class_name] = distribution.get(class_name, 0) + 1
        return distribution


# ============================ 特征提取模型 ============================
class EuroSATFeatureExtractor(nn.Module):
    """EuroSAT特征提取CNN模型"""

    def __init__(self, config):
        super(EuroSATFeatureExtractor, self).__init__()
        self.config = config

        # 卷积层
        self.conv_layers = nn.Sequential(
            # 第一层
            nn.Conv2d(config.num_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第二层
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第三层
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第四层
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, config.feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(config.feature_dim, config.num_classes)
        )

        # 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, config.feature_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, extract_features=False):
        """前向传播

        Args:
            x: 输入图像
            extract_features: 是否提取特征
        Returns:
            如果extract_features=True: 返回特征向量
            否则: 返回分类logits和特征向量
        """
        # 卷积特征
        conv_features = self.conv_layers(x)

        if extract_features:
            # 仅提取特征
            features = self.feature_extractor(conv_features)
            return features
        else:
            # 分类和特征都返回
            features = self.feature_extractor(conv_features)
            logits = self.classifier(features)
            return logits, features


# ============================ 训练器类 ============================
class EuroSATTrainer:
    """EuroSAT模型训练器"""

    def __init__(self, config, model, train_loader, val_loader):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 设备设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        self.model.to(self.device)

        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )

        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        # 早停
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.config.epochs} [训练]')
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            logits, _ = self.model(images)
            loss = self.criterion(logits, labels)

            # 反向传播
            loss.backward()
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

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def validate(self, epoch):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch + 1}/{self.config.epochs} [验证]')
            for batch_idx, batch in enumerate(pbar):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                # 前向传播
                logits, _ = self.model(images)
                loss = self.criterion(logits, labels)

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

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def train(self):
        """完整训练过程"""
        print("开始训练模型...")

        for epoch in range(self.config.epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)

            # 验证
            val_loss, val_acc = self.validate(epoch)

            # 更新学习率
            self.scheduler.step(val_loss)

            # 保存历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            print(f"Epoch {epoch + 1}/{self.config.epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # 早停和保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()

                # 保存最佳模型
                model_path = os.path.join(self.config.output_dir, "models", "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.best_model_state,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                }, model_path)
                print(f"保存最佳模型到: {model_path}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    print(f"早停触发，在epoch {epoch + 1}")
                    break

        # 加载最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        # 保存最终模型
        final_model_path = os.path.join(self.config.output_dir, "models", "final_model.pth")
        torch.save(self.model.state_dict(), final_model_path)

        # 保存训练历史
        history_path = os.path.join(self.config.output_dir, "models", "training_history.pkl")
        with open(history_path, 'wb') as f:
            pickle.dump(self.history, f)

        print(f"训练完成，模型保存到: {final_model_path}")

        return self.history


# ============================ 特征提取器 ============================
class FeatureExtractor:
    """特征提取器"""

    def __init__(self, config, model_path):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载模型
        self.model = EuroSATFeatureExtractor(config)

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"加载模型: {model_path}")
        else:
            print(f"警告: 模型文件不存在 {model_path}，使用随机初始化的模型")

        self.model.to(self.device)
        self.model.eval()

    def extract_features(self, data_loader):
        """从数据集中提取特征"""
        all_features = []
        all_labels = []
        all_paths = []
        all_class_names = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="提取特征"):
                images = batch['image'].to(self.device)

                # 提取特征
                features = self.model(images, extract_features=True)

                # 收集数据
                all_features.append(features.cpu().numpy())
                all_labels.append(batch['label'].numpy())
                all_paths.extend(batch['path'])
                all_class_names.extend(batch['class_name'])

        # 合并所有批次
        all_features = np.vstack(all_features)
        all_labels = np.concatenate(all_labels)

        return {
            'features': all_features,
            'labels': all_labels,
            'paths': all_paths,
            'class_names': all_class_names
        }


# ============================ 可视化工具 ============================
class Visualizer:
    """可视化工具类"""

    @staticmethod
    def plot_training_history(history, save_path):
        """绘制训练历史"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # 损失曲线
        axes[0].plot(history['train_loss'], label='训练损失')
        axes[0].plot(history['val_loss'], label='验证损失')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('损失')
        axes[0].set_title('训练和验证损失')
        axes[0].legend()
        axes[0].grid(True)

        # 准确率曲线
        axes[1].plot(history['train_acc'], label='训练准确率')
        axes[1].plot(history['val_acc'], label='验证准确率')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('准确率 (%)')
        axes[1].set_title('训练和验证准确率')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"训练历史图保存到: {save_path}")

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"混淆矩阵保存到: {save_path}")

    @staticmethod
    def plot_feature_tsne(features, labels, class_names, save_path):
        """使用t-SNE可视化特征"""
        try:
            from sklearn.manifold import TSNE

            # 随机采样部分数据（为了加速）
            n_samples = min(1000, len(features))
            indices = np.random.choice(len(features), n_samples, replace=False)
            features_sample = features[indices]
            labels_sample = labels[indices]

            # 执行t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            features_2d = tsne.fit_transform(features_sample)

            # 绘制散点图
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                                  c=labels_sample, cmap='tab20', alpha=0.6)

            # 添加图例
            handles, _ = scatter.legend_elements()
            legend_labels = [class_names[i] for i in range(len(class_names))]
            plt.legend(handles, legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.xlabel('t-SNE特征1')
            plt.ylabel('t-SNE特征2')
            plt.title('特征空间可视化 (t-SNE)')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"特征可视化保存到: {save_path}")

        except ImportError:
            print("需要安装scikit-learn以使用t-SNE可视化")


# ============================ 主程序 ============================
def main():
    """主函数"""
    print("=" * 60)
    print("EuroSAT卫星图像特征提取系统")
    print("=" * 60)

    # 初始化配置
    config = Config()
    print(f"数据路径: {config.data_root}")
    print(f"输出目录: {config.output_dir}")
    print(f"类别数量: {config.num_classes}")

    # 检查数据目录
    if not os.path.exists(config.data_root):
        print(f"错误: 数据目录不存在 - {config.data_root}")
        return

    # 1. 创建数据集
    print("\n1. 创建数据集...")

    # 检查实际存在的文件夹
    actual_folders = [f for f in os.listdir(config.data_root)
                      if os.path.isdir(os.path.join(config.data_root, f))]
    print(f"实际存在的文件夹: {actual_folders}")

    # 使用实际存在的文件夹作为类别
    actual_class_names = sorted(actual_folders)
    config.num_classes = len(actual_class_names)
    config.class_names = actual_class_names

    print(f"使用类别: {actual_class_names}")
    print(f"类别数量: {config.num_classes}")

    # 创建数据集
    train_dataset = EuroSATDataset(
        root_dir=config.data_root,
        class_names=actual_class_names,
        split='train',
        test_size=0.3
    )

    val_dataset = EuroSATDataset(
        root_dir=config.data_root,
        class_names=actual_class_names,
        split='val',
        test_size=0.3
    )

    test_dataset = EuroSATDataset(
        root_dir=config.data_root,
        class_names=actual_class_names,
        split='test',
        test_size=0.3
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )

    print(f"训练集: {len(train_dataset)} 个样本")
    print(f"验证集: {len(val_dataset)} 个样本")
    print(f"测试集: {len(test_dataset)} 个样本")

    # 2. 训练模型
    print("\n2. 训练特征提取模型...")

    model = EuroSATFeatureExtractor(config)
    trainer = EuroSATTrainer(config, model, train_loader, val_loader)
    history = trainer.train()

    # 3. 可视化训练历史
    print("\n3. 可视化训练结果...")
    visualizer = Visualizer()

    # 绘制训练历史
    history_plot_path = os.path.join(config.output_dir, "visualizations", "training_history.png")
    visualizer.plot_training_history(history, history_plot_path)

    # 4. 提取特征
    print("\n4. 提取特征...")

    # 加载最佳模型
    best_model_path = os.path.join(config.output_dir, "models", "best_model.pth")
    extractor = FeatureExtractor(config, best_model_path)

    # 为所有数据集提取特征
    print("提取训练集特征...")
    train_features = extractor.extract_features(train_loader)

    print("提取验证集特征...")
    val_features = extractor.extract_features(val_loader)

    print("提取测试集特征...")
    test_features = extractor.extract_features(test_loader)

    # 5. 保存特征
    print("\n5. 保存特征...")

    # 保存训练特征
    train_feature_path = os.path.join(config.output_dir, "features", "train_features.pkl")
    with open(train_feature_path, 'wb') as f:
        pickle.dump(train_features, f)
    print(f"训练特征保存到: {train_feature_path}")

    # 保存验证特征
    val_feature_path = os.path.join(config.output_dir, "features", "val_features.pkl")
    with open(val_feature_path, 'wb') as f:
        pickle.dump(val_features, f)
    print(f"验证特征保存到: {val_feature_path}")

    # 保存测试特征
    test_feature_path = os.path.join(config.output_dir, "features", "test_features.pkl")
    with open(test_feature_path, 'wb') as f:
        pickle.dump(test_features, f)
    print(f"测试特征保存到: {test_feature_path}")

    # 6. 评估模型
    print("\n6. 评估模型...")

    # 在测试集上评估
    extractor.model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="测试评估"):
            images = batch['image'].to(extractor.device)
            labels = batch['label'].numpy()

            logits, _ = extractor.model(images)
            _, preds = torch.max(logits, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels)

    # 计算指标
    test_accuracy = accuracy_score(all_labels, all_preds)
    print(f"测试集准确率: {test_accuracy:.4f}")

    # 分类报告
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=actual_class_names))

    # 绘制混淆矩阵
    cm_path = os.path.join(config.output_dir, "visualizations", "confusion_matrix.png")
    visualizer.plot_confusion_matrix(all_labels, all_preds, actual_class_names, cm_path)

    # 7. 特征可视化
    print("\n7. 特征可视化...")

    # 合并所有特征进行t-SNE可视化
    all_features = np.vstack([train_features['features'], val_features['features'], test_features['features']])
    all_labels_combined = np.concatenate([train_features['labels'], val_features['labels'], test_features['labels']])

    tsne_path = os.path.join(config.output_dir, "visualizations", "tsne_visualization.png")
    visualizer.plot_feature_tsne(all_features, all_labels_combined, actual_class_names, tsne_path)

    # 8. 保存配置和元数据
    print("\n8. 保存配置和元数据...")

    metadata = {
        'config': {
            'data_root': config.data_root,
            'num_classes': config.num_classes,
            'class_names': actual_class_names,
            'feature_dim': config.feature_dim,
            'img_size': config.img_size
        },
        'dataset_sizes': {
            'train': len(train_dataset),
            'val': len(val_dataset),
            'test': len(test_dataset)
        },
        'performance': {
            'test_accuracy': float(test_accuracy)
        },
        'training_info': {
            'epochs_trained': len(history['train_loss']),
            'best_val_loss': float(trainer.best_val_loss),
            'final_train_acc': float(history['train_acc'][-1]),
            'final_val_acc': float(history['val_acc'][-1])
        }
    }

    metadata_path = os.path.join(config.output_dir, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"元数据保存到: {metadata_path}")

    print("\n" + "=" * 60)
    print("EuroSAT数据处理完成!")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  模型: {config.output_dir}/models/")
    print(f"  特征: {config.output_dir}/features/")
    print(f"  可视化: {config.output_dir}/visualizations/")
    print(f"  元数据: {metadata_path}")


# ============================ 特征使用示例 ============================
def load_eurosat_features(feature_path):
    """加载提取的特征"""
    with open(feature_path, 'rb') as f:
        features_data = pickle.load(f)
    return features_data


def get_feature_extractor(config_path):
    """获取特征提取器用于智能体系统"""
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    config = Config()
    config.num_classes = metadata['config']['num_classes']
    config.class_names = metadata['config']['class_names']
    config.feature_dim = metadata['config']['feature_dim']

    # 加载模型
    model_path = os.path.join(os.path.dirname(config_path), "..", "models", "best_model.pth")
    extractor = FeatureExtractor(config, model_path)

    return extractor


if __name__ == "__main__":
    # 运行主程序
    main()
