#!/usr/bin/env python
# @Time    : 2020/7/8 16:07
# @Author  : wb
# @File    : main.py

'''
主程序
包括训练，测试等功能
'''
import os
import sys
import h5py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import copy
import time

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"项目根目录: {project_root}")
print(f"Python路径: {sys.path}")

# 导入配置
try:
    from config import opt

    print("✓ 成功导入配置")
except ImportError as e:
    print(f"配置导入错误: {e}")
    sys.exit(1)

# 导入数据模块
try:
    from data.dataset import CSVDataset

    print("✓ 成功导入数据模块")
except ImportError as e:
    print(f"数据模块导入错误: {e}")
    sys.exit(1)

# 定义模型（如果导入失败）
try:
    from models.cnn import SimpleCNN

    print("✓ 成功导入模型")
except ImportError as e:
    print(f"模型导入错误: {e}，将在main.py中直接定义模型")


    # 在main.py中直接定义模型
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=4, input_length=2048):
            super(SimpleCNN, self).__init__()

            print(f"创建SimpleCNN模型: 输入长度={input_length}, 类别数={num_classes}")

            self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm1d(64)
            self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm1d(128)
            self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm1d(256)

            self.pool = nn.MaxPool1d(2)
            self.dropout = nn.Dropout(0.5)

            # 计算全连接层输入尺寸
            fc_input_size = 256 * (input_length // 8)

            self.fc1 = nn.Linear(fc_input_size, 512)
            self.fc2 = nn.Linear(512, num_classes)

            # 用于特征提取的中间层
            self.feature = None

        def forward(self, x):
            # 卷积层
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.pool(x)

            # 展平
            x = x.view(x.size(0), -1)

            # 保存中间特征
            self.feature = x.detach().cpu().numpy()

            # 全连接层
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)

            return x

        def save(self, name):
            """保存模型"""
            model_path = f'results/{name}_model.pth'
            os.makedirs('results', exist_ok=True)
            torch.save(self.state_dict(), model_path)
            return model_path


def train():
    '''
    训练模块
    '''
    print("开始训练...")

    # step1: 加载训练数据并获取数据信息
    try:
        train_data = CSVDataset(opt.train_data_root, train=True)
        val_data = CSVDataset(opt.val_data_root, train=False)

        # 获取特征维度和类别数
        feature_dim = train_data.get_feature_dim()
        num_classes = train_data.get_num_classes()

        print(f"特征维度: {feature_dim}, 类别数: {num_classes}")
    except Exception as e:
        print(f"数据加载错误: {e}")
        return

    # step2: 创建模型
    model = SimpleCNN(num_classes=num_classes, input_length=feature_dim)
    if opt.use_gpu:
        model.cuda()
        print("使用GPU进行训练")
    else:
        print("使用CPU进行训练")

    # step3: 数据加载器
    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_data, opt.batch_size, shuffle=False, num_workers=0)

    print(f"训练集大小: {len(train_data)}, 验证集大小: {len(val_data)}")

    # step4: 目标函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_decay_iters, opt.lr_decay)

    # step5: 设置TensorBoard和训练记录
    writer = SummaryWriter('logs')
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    device = torch.device(opt.device)

    # step6: 训练循环
    for epoch in range(opt.max_epoch):
        start_time = time.time()
        print(f'Starting epoch {epoch + 1} / {opt.max_epoch}')

        # 设置为训练模式
        model.train()
        running_loss = 0.0

        for ii, (data, label) in enumerate(train_dataloader):
            # 添加通道维度 [batch, 1, feature_length]
            data = data.unsqueeze(1)
            data, label = data.float(), label.long()
            input, target = data.to(device), label.to(device)

            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 打印训练信息
            if (ii + 1) % opt.print_every == 0:
                avg_loss = running_loss / opt.print_every
                print(f'Epoch {epoch + 1}, Batch {ii + 1}, Loss: {avg_loss:.4f}')
                writer.add_scalar('train_loss', avg_loss, epoch * len(train_dataloader) + ii)
                running_loss = 0.0

        # 更新学习率
        scheduler.step()

        # 计算训练和验证准确率
        train_acc, _ = check_accuracy(model, train_dataloader, device)
        val_acc, _ = check_accuracy(model, val_dataloader, device)

        # 记录到TensorBoard
        writer.add_scalars('accuracy', {'train': train_acc, 'val': val_acc}, epoch)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f'新的最佳验证准确率: {best_acc:.4f}')

        end_time = time.time()
        epoch_time = end_time - start_time
        print(
            f'Epoch {epoch + 1} 完成, 时间: {epoch_time:.2f}秒, 训练准确率: {train_acc:.4f}, 验证准确率: {val_acc:.4f}')
        print('-' * 50)

    # 训练结束，加载最佳模型
    model.load_state_dict(best_model_wts)

    # 最终验证
    val_acc, confuse_matrix = check_accuracy(model, val_dataloader, device, error_analysis=True)

    # 保存混淆矩阵
    data_pd = pd.DataFrame(confuse_matrix)
    data_pd.to_excel(opt.result_file, index=True)
    print(f'混淆矩阵已保存到: {opt.result_file}')

    # 保存模型
    model_save_path = model.save(opt.model)
    print(f'最优模型保存在: {model_save_path}')

    writer.close()
    print("训练完成！")


def test():
    '''
    测试模块
    '''
    print("开始测试...")

    # step1: 加载测试数据
    try:
        test_dataset = CSVDataset(opt.test_data_root, train=False)
        test_loader = DataLoader(test_dataset, opt.batch_size, shuffle=False, num_workers=0)

        # 获取类别数
        num_classes = test_dataset.get_num_classes()
        feature_dim = test_dataset.get_feature_dim()

        print(f'测试集大小: {len(test_dataset)}')
        print(f'特征维度: {feature_dim}, 类别数: {num_classes}')
    except Exception as e:
        print(f"测试数据加载错误: {e}")
        return

    # step2: 加载模型
    model = SimpleCNN(num_classes=num_classes, input_length=feature_dim).eval()

    if opt.load_model_path and os.path.exists(opt.load_model_path):
        try:
            model.load_state_dict(torch.load(opt.load_model_path))
            print(f'已加载预训练模型: {opt.load_model_path}')
        except Exception as e:
            print(f"加载预训练模型失败: {e}")

    # step3: 设备设置
    device = torch.device(opt.device)
    model = model.to(device)

    # step4: 测试
    try:
        f = h5py.File(opt.feature_filename, 'w')
        f.create_dataset('y_true', data=test_dataset.y)

        test_acc, confuse_matrix = check_accuracy(model, test_loader, device, feature_file=f, error_analysis=True)

        f.close()

        # 保存测试结果
        test_result_file = 'results/test_confusion_matrix.xlsx'
        pd.DataFrame(confuse_matrix).to_excel(test_result_file, index=True)
        print(f'测试混淆矩阵已保存到: {test_result_file}')
        print(f'测试准确率: {test_acc:.4f}')
    except Exception as e:
        print(f"测试过程错误: {e}")


def check_accuracy(model, loader, device, feature_file=None, error_analysis=False):
    '''
    检查模型的准确率
    '''
    model.eval()

    all_preds = []
    all_labels = []
    num_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.unsqueeze(1)
            x, y = x.float(), y.long()
            x, y = x.to(device), y.to(device)

            scores = model(x)
            preds = scores.max(1, keepdim=True)[1]

            num_correct += preds.eq(y.view_as(preds)).sum().item()
            total_samples += y.size(0)

            if error_analysis:
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

            if feature_file is not None and hasattr(model, 'feature'):
                feature_output = model.feature
                if 'X_features' not in feature_file:
                    feature_file.create_dataset('X_features', data=feature_output,
                                                maxshape=(None, feature_output.shape[1]))
                else:
                    # 扩展数据集
                    feature_file['X_features'].resize((feature_file['X_features'].shape[0] + feature_output.shape[0]),
                                                      axis=0)
                    feature_file['X_features'][-feature_output.shape[0]:] = feature_output

    acc = float(num_correct) / total_samples if total_samples > 0 else 0

    if error_analysis and all_preds and all_labels:
        try:
            confuse_matrix = pd.crosstab(
                pd.Series(all_preds).squeeze(),
                pd.Series(all_labels),
                margins=True
            )
        except Exception as e:
            print(f"创建混淆矩阵错误: {e}")
            confuse_matrix = None
    else:
        confuse_matrix = None

    print(f'准确率: {num_correct}/{total_samples} ({100 * acc:.2f}%)')
    return acc, confuse_matrix


if __name__ == '__main__':
    # 根据需求选择训练或测试
    train()  # 取消注释以训练
    # test()   # 取消注释以测试

