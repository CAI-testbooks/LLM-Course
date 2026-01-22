# csv_to_h5.py
import pandas as pd
import h5py
import numpy as np
import os


def csv_to_h5(csv_path, h5_path, is_train=True):
    """
    将CSV文件转换为h5格式
    """
    print(f"正在转换 {csv_path} 到 {h5_path}")

    # 读取CSV文件
    df = pd.read_csv(csv_path)

    # 假设最后一列是标签，其他列是特征
    features = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values

    # 创建h5文件
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('data', data=features)
        f.create_dataset('label', data=labels)

    print(f"转换完成！特征形状: {features.shape}, 标签形状: {labels.shape}")

    # 统计类别信息
    unique_labels = np.unique(labels)
    print(f"类别数量: {len(unique_labels)}, 类别标签: {unique_labels}")

    return len(unique_labels)


# 转换训练集和测试集
train_csv = r"C:\Users\29514\Desktop\TrainData_50%Test.CSV"
test_csv = r"C:\Users\29514\Desktop\TestData_50%Test_v2.CSV"

train_h5 = "train_data.h5"
test_h5 = "test_data.h5"

# 执行转换
num_classes_train = csv_to_h5(train_csv, train_h5, is_train=True)
num_classes_test = csv_to_h5(test_csv, test_h5, is_train=False)

print(f"训练集类别数: {num_classes_train}")
print(f"测试集类别数: {num_classes_test}")
