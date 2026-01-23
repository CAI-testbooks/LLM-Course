# debug_model.py
import torch
import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import opt
from data.dataset import CSVDataset


def debug_model_flow():
    """调试模型数据流"""
    print("调试模型数据流...")

    # 加载少量数据
    try:
        train_data = CSVDataset(opt.train_data_root, train=True)
        feature_dim = train_data.get_feature_dim()
        num_classes = train_data.get_num_classes()

        print(f"数据特征维度: {feature_dim}")
        print(f"类别数量: {num_classes}")

        # 获取一个batch的数据
        from torch.utils.data import DataLoader
        dataloader = DataLoader(train_data, batch_size=2, shuffle=False)
        sample_batch = next(iter(dataloader))
        data, labels = sample_batch
        data = data.unsqueeze(1)  # 添加通道维度

        print(f"输入数据形状: {data.shape}")

        # 手动模拟卷积层计算
        print("\n手动模拟卷积层计算:")

        # 卷积层1
        conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        x = F.relu(conv1(data))
        print(f"卷积层1后: {x.shape}")

        # 池化1
        pool = nn.MaxPool1d(2)
        x = pool(x)
        print(f"池化1后: {x.shape}")

        # 卷积层2
        conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        x = F.relu(conv2(x))
        print(f"卷积层2后: {x.shape}")

        # 池化2
        x = pool(x)
        print(f"池化2后: {x.shape}")

        # 卷积层3
        conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        x = F.relu(conv3(x))
        print(f"卷积层3后: {x.shape}")

        # 池化3
        x = pool(x)
        print(f"池化3后: {x.shape}")

        # 展平
        x = x.view(x.size(0), -1)
        print(f"展平后: {x.shape}")

        # 计算期望的全连接层输入尺寸
        expected_fc_input = 256 * (feature_dim // 8)
        print(f"期望的全连接层输入尺寸: {expected_fc_input}")
        print(f"实际的全连接层输入尺寸: {x.shape[1]}")

    except Exception as e:
        print(f"调试错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_model_flow()