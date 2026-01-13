import numpy as np
import pandas as pd
from scipy.io import loadmat
from joblib import dump, load
import sklearn
import torch

# ------------------------ 1. 读取MAT文件（10个类别，对应驱动端数据） ------------------------
file_names = ['0_0.mat','7_1.mat','7_2.mat','7_3.mat','14_1.mat','14_2.mat','14_3.mat','21_1.mat','21_2.mat','21_3.mat']
# 驱动端振动数据列名（对应10个MAT文件）
data_columns = ['X097_DE_time', 'X105_DE_time', 'X118_DE_time', 'X130_DE_time', 'X169_DE_time',
                'X185_DE_time','X197_DE_time','X209_DE_time','X222_DE_time','X234_DE_time']
# 10分类标签名
columns_name = ['de_normal', 'de_7_inner', 'de_7_ball', 'de_7_outer', 
                'de_14_inner', 'de_14_ball', 'de_14_outer', 
                'de_21_inner', 'de_21_ball', 'de_21_outer']

# 读取并合并10类数据
data_12k_10c = pd.DataFrame()
for index in range(10):
    data = loadmat(f'matfiles\\{file_names[index]}')  # 确保matfiles文件夹与代码同目录
    dataList = data[data_columns[index]].reshape(-1)  # 展平为一维时间序列
    # 统一数据长度（取最小长度119808，确保所有类别数据长度一致）
    data_12k_10c[columns_name[index]] = dataList[:119808]

print(f"原始数据形状: {data_12k_10c.shape}")  # 应输出 (119808, 10)
data_12k_10c.to_csv('data_12k_10c.csv', index=False)  # 保存原始10分类数据


# ------------------------ 2. 数据预处理工具函数 ------------------------
def split_data_with_overlap(data, time_steps, label, overlap_ratio=0.5):
    """重叠窗口切分（参考论文方法）
    参数：data-一维时间序列，time_steps-窗口长度，label-类别标签，overlap_ratio-重叠率
    返回：切分后的样本（每行为1个样本，最后一列为标签）
    """
    stride = int(time_steps * (1 - overlap_ratio))  # 步长=512（重叠率0.5）
    samples = (len(data) - time_steps) // stride + 1  # 每类样本数=233
    data_list = []
    for i in range(samples):
        start_idx = i * stride
        end_idx = start_idx + time_steps
        temp_data = data[start_idx:end_idx].tolist()
        temp_data.append(label)  # 末尾添加类别标签
        data_list.append(temp_data)
    return pd.DataFrame(data_list, columns=[x for x in range(time_steps + 1)])

def normalize(data):
    """(0,1)归一化（统一数据尺度，修正：启用归一化）"""
    return (data - min(data)) / (max(data) - min(data))


# ------------------------ 3. 制作训练/验证/测试集 ------------------------
def make_datasets(data_file_csv, split_rate=[0.7, 0.2, 0.1]):
    """
    生成10分类数据集并划分
    参数：data_file_csv-原始数据（data_12k_10c.csv），split_rate-划分比例
    返回：训练集/验证集/测试集（joblib格式）
    """
    # 读取原始数据
    origin_data = pd.read_csv(data_file_csv)
    time_steps = 1024  # 窗口长度
    overlap_ratio = 0.5  # 重叠率
    samples_data = pd.DataFrame(columns=[x for x in range(time_steps + 1)])
    label = 0  # 类别标签（0-9对应10类）

    # 遍历每类数据，切分+归一化
    for column_name, column_data in origin_data.items():
        column_data = normalize(column_data)  # 修正：启用归一化
        split_data = split_data_with_overlap(column_data, time_steps, label, overlap_ratio)
        samples_data = pd.concat([samples_data, split_data])
        label += 1

    # 打乱样本（保证类别分布均匀）
    samples_data = sklearn.utils.shuffle(samples_data, random_state=42)
    print(f"总样本数: {samples_data.shape[0]}")  # 应输出 2330（10类×233样本）

    # 划分训练/验证/测试集
    sample_len = len(samples_data)
    train_len = int(sample_len * split_rate[0])  # 1631个训练样本
    val_len = int(sample_len * split_rate[1])    # 466个验证样本
    train_set = samples_data.iloc[0:train_len, :]
    val_set = samples_data.iloc[train_len:train_len + val_len, :]
    test_set = samples_data.iloc[train_len + val_len:sample_len, :]

    # 保存数据集（按文件一命名规则）
    dump(train_set, 'train_set')
    dump(val_set, 'val_set')
    dump(test_set, 'test_set')
    return train_set, val_set, test_set, samples_data


# ------------------------ 4. 转换为PyTorch张量 ------------------------
def make_data_labels(dataframe):
    """将DataFrame转换为模型可处理的张量"""
    x_data = dataframe.iloc[:, 0:-1]  # 特征（前1024列）
    y_label = dataframe.iloc[:, -1]   # 标签（最后1列）
    x_data = torch.tensor(x_data.values).float()
    y_label = torch.tensor(y_label.values.astype('int64'))  # 标签为int64（PyTorch分类要求）
    return x_data, y_label


# ------------------------ 执行流程 ------------------------
# 第一步：生成训练/验证/测试集（DataFrame格式）
train_set, val_set, test_set, samples_data = make_datasets('data_12k_10c.csv')

# 第二步：转换为张量并保存
train_xdata, train_ylabel = make_data_labels(train_set)
val_xdata, val_ylabel = make_data_labels(val_set)
test_xdata, test_ylabel = make_data_labels(test_set)

dump(train_xdata, 'trainX_1024_10c')
dump(val_xdata, 'valX_1024_10c')
dump(test_xdata, 'testX_1024_10c')
dump(train_ylabel, 'trainY_1024_10c')
dump(val_ylabel, 'valY_1024_10c')
dump(test_ylabel, 'testY_1024_10c')

# 打印最终张量形状（验证正确性）
print("\n张量数据形状:")
print(f"训练集 X: {train_xdata.shape}, Y: {train_ylabel.shape}")  # (1631,1024), (1631,)
print(f"验证集 X: {val_xdata.shape}, Y: {val_ylabel.shape}")      # (466,1024), (466,)
print(f"测试集 X: {test_xdata.shape}, Y: {test_ylabel.shape}")    # (233,1024), (233,)