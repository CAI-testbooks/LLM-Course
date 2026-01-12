# config.py
import torch
import os


class Config:
    # 数据路径
    train_data_root = r'C:\Users\29514\Desktop\TrainData_50%Test.CSV'
    val_data_root = r'C:\Users\29514\Desktop\TestData_50%Test_v2.CSV'
    test_data_root = r'C:\Users\29514\Desktop\TestData_50%Test_v2.CSV'

    # 模型配置 - 使用 SimpleCNN 或 CWRUModel
    model = 'AdaptiveCNN'  # 或者 'CWRUModel'
    load_model_path = None

    # 训练参数
    batch_size = 32
    max_epoch = 10
    lr = 0.001
    weight_decay = 1e-4
    lr_decay_iters = 30
    lr_decay = 0.1
    print_every = 10

    # 设备配置
    use_gpu = torch.cuda.is_available()
    device = 'cuda:0' if use_gpu else 'cpu'

    # 结果保存
    result_file = 'results/confusion_matrix.xlsx'
    feature_filename = 'results/features.h5'

    def __init__(self):
        os.makedirs('results', exist_ok=True)
        os.makedirs('logs', exist_ok=True)


opt = Config()