# import os
# import numpy as np
# import torch
# from torch.utils.data import DataLoader, DistributedSampler, Subset
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# from Dataset.HCPDataset import create_dataloader
# from Model.utils.config import Config
# from tqdm import tqdm
# import time
# import matplotlib.pyplot as plt
# from torch.optim.lr_scheduler import StepLR
# from torch.nn import L1Loss
# from Model.STEncoder import STEncoder
# from Model.STDecoder import STDecoder
# import math
#
#
# MASK_RATIO = 0.75
# MASK_WEIGHT = 0.7
# EPOCH_NUM = 2
# LR = 1e-4
#
#
# def data_read(dir_path):
#     with open(dir_path, "r") as f:
#         raw_data = f.read()
#         data = raw_data[1:-2].split(", ")
#     return np.asarray(data, float)
#
#
# def draw_loss(loss_path, draw_path, phase):
#     y_train_loss = data_read(loss_path)
#     x_train_loss = range(len(y_train_loss))
#     plt.figure()
#     ax = plt.axes()
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     plt.xlabel('iters')
#     plt.ylabel('loss')
#     plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="loss")
#     plt.xticks(x_train_loss)
#     plt.legend()
#     plt.title(phase + ' Loss')
#     plt.savefig(draw_path)
#
#
# def zscore_and_remove_nan(batch_data):
#     batch_data_cpu = batch_data.cpu().numpy()
#
#     batch_data_cpu = np.nan_to_num(batch_data_cpu, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
#
#     mean = np.nanmean(batch_data_cpu, axis=1, keepdims=True)
#     std = np.nanstd(batch_data_cpu, axis=1, keepdims=True)
#
#     zscore_data = (batch_data_cpu - mean) / (std + 1e-6)  # 避免除以 0
#
#     zscore_data = np.nan_to_num(zscore_data, nan=0.0)
#
#     return torch.tensor(zscore_data, dtype=torch.float32).to(batch_data.device)
#
#
# def compute_weighted_mse_loss(x, reconstructed_x, mask, mask_weight=5.0):
#     masked_loss = ((x - reconstructed_x) ** 2) * mask
#     masked_loss = masked_loss.sum() / mask.sum().clamp(min=1)
#
#     non_mask = 1 - mask
#     non_masked_loss = ((x - reconstructed_x) ** 2) * non_mask
#     non_masked_loss = non_masked_loss.sum() / non_mask.sum().clamp(min=1)
#
#
#     total_loss = mask_weight * masked_loss + non_masked_loss
#     return total_loss
#
# def count_parameters(model):
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     return total_params, trainable_params
#
#
# def train_one_epoch(train_loader, encoder, encoder_optimizer, decoder, decoder_optimizer):
#     encoder.train()
#     decoder.train()
#     epoch_loss = 0
#
#     pbar = tqdm(train_loader, desc="Training", leave=False)
#     encoder_optimizer.zero_grad()
#     decoder_optimizer.zero_grad()
#     batch_num = 0
#     for count,(surface_data) in enumerate(pbar):
#         batch = surface_data.to("cuda", dtype=torch.float32)
#         x_l = batch[:, :, :642]
#         x_r = batch[:, :, 2562:3204]
#         batch = torch.cat((x_l, x_r), dim=-1)
#         batch =zscore_and_remove_nan(batch_data=batch)
#
#
#         fMRI_embed, mask = encoder.forward_phase1(batch, mask_ratio=MASK_RATIO)
#         reconstructed = decoder.reconstruct_phase1(fMRI_embed)
#         loss = compute_weighted_mse_loss(batch, reconstructed, mask,mask_weight=MASK_WEIGHT)
#         loss.backward()
#         encoder_optimizer.step()
#         encoder_optimizer.zero_grad()
#         decoder_optimizer.step()
#         decoder_optimizer.zero_grad()
#
#         epoch_loss += loss.item()
#         batch_num = batch_num + 1
#     epoch_loss = epoch_loss / len(train_loader)
#     return epoch_loss
#
# def validate_one_epoch(valid_loader, encoder,decoder):
#     encoder.eval()
#     decoder.eval()
#     epoch_loss = 0
#     with torch.no_grad():
#
#         pbar = tqdm(valid_loader, desc="Validation", leave=False)
#
#         for batch_idx,(surface_data) in enumerate(pbar):
#             batch = surface_data.to("cuda", dtype=torch.float32)
#             x_l = batch[:, :, :642]
#             x_r = batch[:, :, 2562:3204]
#             batch = torch.cat((x_l, x_r), dim=-1)
#             batch = zscore_and_remove_nan(batch_data=batch)
#
#             fMRI_embed, mask = encoder.forward_phase1(batch, mask_ratio=MASK_RATIO)
#             reconstructed = decoder.reconstruct_phase1(fMRI_embed)
#
#             loss = compute_weighted_mse_loss(batch, reconstructed, mask, mask_weight=MASK_WEIGHT)
#             epoch_loss += loss.item()
#
#     epoch_loss = epoch_loss / len(valid_loader)
#
#     return epoch_loss
#
#
# def train(train_loader, val_loader, encoder, encoder_optimizer, encoder_scheduler,
#             decoder, decoder_optimizer, decoder_scheduler,
#           num_epochs,save_path=""):
#     loss_list = []
#     test_loss_list = []
#     best_loss = float('inf')
#     for epoch in range(num_epochs):
#
#         print(f"Epoch {epoch + 1}/{num_epochs}")
#         start_time = time.time()
#
#         epoch_loss = train_one_epoch(train_loader, encoder, encoder_optimizer, decoder, decoder_optimizer)
#         loss_list.append(epoch_loss)
#
#         epoch_duration = time.time() - start_time
#
#         print(f"Train MSE Loss: {epoch_loss:.4f}, Time: {epoch_duration:.2f} seconds")
#
#         if encoder_scheduler:
#             encoder_scheduler.step()  # 更新学习率
#         if decoder_scheduler:
#             decoder_scheduler.step()  # 更新学习率
#
#
#         start_time = time.time()
#         val_loss = validate_one_epoch(val_loader, encoder,decoder)
#         test_loss_list.append(val_loss)
#
#         epoch_duration = time.time() - start_time
#         print(f"Test MSE Loss: {val_loss:.4f}, Time: {epoch_duration:.2f} seconds")
#
#         if val_loss < best_loss:
#             best_loss = val_loss
#             torch.save(encoder.state_dict(), save_path + '\Model\Phase1_encoder.pth')
#             torch.save(decoder.state_dict(), save_path + '\Model\Phase1_decoder.pth')
#
#             print(f"Model saved with validation loss: {val_loss:.4f}")
#
#
#
# def load_model(model, save_path):
#     checkpoint = torch.load(save_path, weights_only=False)
#     model.load_state_dict(checkpoint)
#     return model
#
# def Phase1():
#     length_list = [20]
#     for len_idx in length_list:
#         data_dir = r"C:\Users\20355\Desktop\CogReader-main\Data\HCP"
#         save_path = r"C:\Users\20355\Desktop\CogReader-main\Results\Training\Phase1"
#         Model_path = save_path + "\Model"
#         os.makedirs(save_path, exist_ok=True)
#         os.makedirs(Model_path, exist_ok=True)
#         num_epochs = EPOCH_NUM
#         learning_rate = LR
#
#         cfg = Config()
#         encoder = STEncoder(cfg,len_idx)
#         encoder = encoder.to("cuda")
#         decoder = STDecoder(cfg,len_idx)
#         decoder = decoder.to("cuda")
#
#         encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
#         decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
#
#
#         print("loading data....")
#         train_data, val_data = create_dataloader(
#             data_dir=data_dir,
#             random_state=39,
#             test_size=0.2,
#             length=len_idx
#         )
#
#         train_loader = DataLoader(train_data, batch_size=2, num_workers=4)
#         val_loader = DataLoader(val_data, batch_size=2,  num_workers=4)
#
#         print("loading data success!!!")
#         print(f"train dataset: {len(train_loader)}")
#         print(f"test dataset: {len(val_loader)}")
#         print(f"Processing length: {len_idx}")
#
#         encoder_scheduler = StepLR(encoder_optimizer, step_size=5, gamma=0.5)
#         decoder_scheduler = StepLR(decoder_optimizer, step_size=5, gamma=0.5)
#
#         train(train_loader, val_loader, encoder, encoder_optimizer, encoder_scheduler,
#               decoder, decoder_optimizer, decoder_scheduler,
#               num_epochs,save_path)
#
#
# if __name__ == '__main__':
#     Phase1()




# 导入必要工具库：操作系统交互、数值计算、PyTorch核心框架、数据加载与分布式训练、
# 模型配置、进度条、时间统计、绘图、学习率调度、损失函数、自定义模型与数据集工具
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler, Subset  # 数据加载相关
import torch.distributed as dist  # 分布式训练预留（文档未使用，保留原结构）
from torch.nn.parallel import DistributedDataParallel as DDP  # 分布式并行预留
from Dataset.HCPDataset import create_dataloader  # 加载HCP数据集（文档Phase1预训练数据集）
from Model.utils.config import Config  # 模型超参数配置类
from tqdm import tqdm  # 训练进度可视化
import time  # 训练时间统计
import matplotlib.pyplot as plt  # 损失曲线绘制
from torch.optim.lr_scheduler import StepLR  # 步长衰减学习率调度器
from torch.nn import L1Loss  # L1损失（预留，文档用加权MSE）
from Model.STEncoder import STEncoder  # 时空编码器（文档：空间+时间Transformer模块，Phase1核心）
from Model.STDecoder import STDecoder  # 时空解码器（文档：MAE解码器，用于重构fMRI信号）
import math


# 文档Phase1预训练核心超参数（对应论文HCP预训练配置）
MASK_RATIO = 0.75  # 随机掩码率：75%（文档明确预训练阶段采用75%掩码率）
MASK_WEIGHT = 0.7  # 掩码区域损失权重（突出掩码部分重构，增强表征学习）
EPOCH_NUM = 2  # 预训练轮数（示例值，文档预训练为20轮，此处保留原代码配置）
LR = 1e-4  # 初始学习率（文档Phase1预训练初始LR=1e-4，与配置一致）


def data_read(dir_path):
    """
    读取损失日志文件数据（用于后续绘制损失曲线）
    输入：dir_path - 损失文件路径
    输出：np.ndarray - 解析后的损失数值数组
    """
    with open(dir_path, "r") as f:
        raw_data = f.read()
        # 解析字符串格式的损失数据（去除首尾多余字符，按逗号分割）
        data = raw_data[1:-2].split(", ")
    # 转换为浮点型numpy数组返回
    return np.asarray(data, float)


def draw_loss(loss_path, draw_path, phase):
    """
    绘制训练/验证损失曲线（监控训练收敛情况）
    输入：
        loss_path - 损失数据文件路径
        draw_path - 图表保存路径
        phase - 阶段标识（"Train"或"Validation"）
    """
    # 读取损失数据
    y_train_loss = data_read(loss_path)
    # 生成x轴（迭代次数）
    x_train_loss = range(len(y_train_loss))
    # 创建图表
    plt.figure()
    ax = plt.axes()
    # 隐藏上、右坐标轴（美化图表）
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 设置坐标轴标签
    plt.xlabel('iters')
    plt.ylabel('loss')
    # 绘制损失曲线
    plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="loss")
    # 设置x轴刻度
    plt.xticks(x_train_loss)
    # 显示图例
    plt.legend()
    # 设置图表标题
    plt.title(phase + ' Loss')
    # 保存图表
    plt.savefig(draw_path)


def zscore_and_remove_nan(batch_data):
    """
    fMRI信号预处理（对应文档数据预处理步骤）：去NaN值 + Z-score标准化
    输入：batch_data - 原始fMRI批量数据（shape: [batch_size, seq_len, num_vertices]）
    输出：torch.Tensor - 预处理后的fMRI数据（同输入shape，已标准化且无NaN）
    """
    # 转移到CPU并转换为numpy数组（便于数值处理）
    batch_data_cpu = batch_data.cpu().numpy()

    # 替换NaN、正无穷、负无穷（避免数据异常影响训练）
    batch_data_cpu = np.nan_to_num(
        batch_data_cpu,
        nan=0.0,
        posinf=np.finfo(np.float32).max,
        neginf=np.finfo(np.float32).min
    )

    # 计算每个样本在时间维度上的均值和标准差（按文档标准化逻辑）
    mean = np.nanmean(batch_data_cpu, axis=1, keepdims=True)
    std = np.nanstd(batch_data_cpu, axis=1, keepdims=True)

    # Z-score标准化：(x - mean) / (std + 1e-6)，加1e-6避免除以0
    zscore_data = (batch_data_cpu - mean) / (std + 1e-6)

    # 再次替换标准化后可能出现的NaN（极端值导致）
    zscore_data = np.nan_to_num(zscore_data, nan=0.0)

    # 转换回Tensor并放回原设备（GPU/CPU）
    return torch.tensor(zscore_data, dtype=torch.float32).to(batch_data.device)


def compute_weighted_mse_loss(x, reconstructed_x, mask, mask_weight=5.0):
    """
    计算加权MSE重构损失（文档MAE核心损失函数，突出掩码区域学习）
    输入：
        x - 原始fMRI数据（shape: [batch_size, seq_len, num_vertices]）
        reconstructed_x - 解码器重构的fMRI数据（同x shape）
        mask - 掩码矩阵（1表示掩码区域，0表示非掩码区域）
        mask_weight - 掩码区域损失权重（放大掩码部分损失，增强表征能力）
    输出：torch.Tensor - 加权后总损失
    """
    # 计算掩码区域损失：仅对被掩码的位置计算MSE，按掩码数量归一化
    masked_loss = ((x - reconstructed_x) ** 2) * mask
    masked_loss = masked_loss.sum() / mask.sum().clamp(min=1)  # clamp避免mask.sum=0导致除以0

    # 计算非掩码区域损失：对未掩码位置计算MSE，按非掩码数量归一化
    non_mask = 1 - mask
    non_masked_loss = ((x - reconstructed_x) ** 2) * non_mask
    non_masked_loss = non_masked_loss.sum() / non_mask.sum().clamp(min=1)

    # 总损失：掩码区域损失×权重 + 非掩码区域损失（文档MAE损失逻辑）
    total_loss = mask_weight * masked_loss + non_masked_loss
    return total_loss


def count_parameters(model):
    """
    统计模型参数总量和可训练参数数量（模型复杂度分析）
    输入：model - PyTorch模型实例
    输出：tuple - (总参数数, 可训练参数数)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def train_one_epoch(train_loader, encoder, encoder_optimizer, decoder, decoder_optimizer):
    """
    训练单轮（对应文档Phase1预训练的单轮训练流程）
    输入：
        train_loader - 训练数据加载器
        encoder - STEncoder实例（时空编码器，文档MAE编码器）
        encoder_optimizer - 编码器优化器
        decoder - STDecoder实例（时空解码器，文档MAE解码器）
        decoder_optimizer - 解码器优化器
    输出：float - 单轮平均训练损失
    """
    # 设置模型为训练模式（启用Dropout、BatchNorm更新等）
    encoder.train()
    decoder.train()
    # 累计单轮损失
    epoch_loss = 0

    # 进度条可视化训练过程
    pbar = tqdm(train_loader, desc="Training", leave=False)
    # 初始化优化器梯度
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    batch_num = 0
    for count,(surface_data) in enumerate(pbar):
        # 读取fMRI数据并转移到GPU，转换为float32类型
        batch = surface_data.to("cuda", dtype=torch.float32)
        # 提取左右脑皮层顶点数据（文档fMRI数据为皮层顶点神经活动，左右脑分离存储）
        x_l = batch[:, :, :642]  # 左脑皮层顶点（前642个维度）
        x_r = batch[:, :, 2562:3204]  # 右脑皮层顶点（2562-3204维度）
        # 拼接左右脑数据（形成完整的皮层顶点特征，维度：642+642=1284）
        batch = torch.cat((x_l, x_r), dim=-1)
        # fMRI数据预处理：去NaN + Z-score标准化（文档强制要求的预处理步骤）
        batch = zscore_and_remove_nan(batch_data=batch)

        # 文档MAE核心流程1：编码器掩码编码
        # 输入原始fMRI数据，生成掩码后的fMRI嵌入 + 掩码矩阵
        fMRI_embed, mask = encoder.forward_phase1(batch, mask_ratio=MASK_RATIO)
        # 文档MAE核心流程2：解码器重构
        # 输入fMRI嵌入，重构原始fMRI信号
        reconstructed = decoder.reconstruct_phase1(fMRI_embed)
        # 计算加权MSE重构损失（文档Phase1预训练损失函数）
        loss = compute_weighted_mse_loss(batch, reconstructed, mask, mask_weight=MASK_WEIGHT)
        # 反向传播：计算梯度
        loss.backward()
        # 更新编码器参数
        encoder_optimizer.step()
        encoder_optimizer.zero_grad()  # 重置编码器梯度
        # 更新解码器参数
        decoder_optimizer.step()
        decoder_optimizer.zero_grad()  # 重置解码器梯度

        # 累计损失
        epoch_loss += loss.item()
        batch_num = batch_num + 1
    # 计算单轮平均损失（按训练样本数归一化）
    epoch_loss = epoch_loss / len(train_loader)
    return epoch_loss


def validate_one_epoch(valid_loader, encoder, decoder):
    """
    验证单轮（对应文档Phase1预训练的验证流程，无梯度更新）
    输入：
        valid_loader - 验证数据加载器
        encoder - STEncoder实例（时空编码器）
        decoder - STDecoder实例（时空解码器）
    输出：float - 单轮平均验证损失
    """
    # 设置模型为评估模式（禁用Dropout、固定BatchNorm等）
    encoder.eval()
    decoder.eval()
    # 累计单轮验证损失
    epoch_loss = 0
    # 禁用梯度计算（节省显存，加速验证）
    with torch.no_grad():
        # 进度条可视化验证过程
        pbar = tqdm(valid_loader, desc="Validation", leave=False)

        for batch_idx,(surface_data) in enumerate(pbar):
            # 数据读取、转移设备、左右脑拼接（同训练流程）
            batch = surface_data.to("cuda", dtype=torch.float32)
            x_l = batch[:, :, :642]
            x_r = batch[:, :, 2562:3204]
            batch = torch.cat((x_l, x_r), dim=-1)
            # fMRI数据预处理（同训练流程）
            batch = zscore_and_remove_nan(batch_data=batch)

            # MAE编码-重构流程（同训练，无梯度更新）
            fMRI_embed, mask = encoder.forward_phase1(batch, mask_ratio=MASK_RATIO)
            reconstructed = decoder.reconstruct_phase1(fMRI_embed)

            # 计算验证损失
            loss = compute_weighted_mse_loss(batch, reconstructed, mask, mask_weight=MASK_WEIGHT)
            epoch_loss += loss.item()

    # 计算单轮平均验证损失
    epoch_loss = epoch_loss / len(valid_loader)
    return epoch_loss


def train(train_loader, val_loader, encoder, encoder_optimizer, encoder_scheduler,
          decoder, decoder_optimizer, decoder_scheduler,
          num_epochs, save_path=""):
    """
    文档Phase1预训练主流程（多轮训练+验证+模型保存+学习率调度）
    输入：
        train_loader/val_loader - 训练/验证数据加载器
        encoder/decoder - 编码器/解码器实例
        encoder_optimizer/decoder_optimizer - 编码器/解码器优化器
        encoder_scheduler/decoder_scheduler - 编码器/解码器学习率调度器
        num_epochs - 预训练总轮数
        save_path - 模型和结果保存路径
    """
    # 记录训练/验证损失（用于后续绘图）
    loss_list = []
    test_loss_list = []
    # 初始化最优验证损失（用于保存最优模型）
    best_loss = float('inf')
    for epoch in range(num_epochs):
        # 打印当前轮次信息
        print(f"Epoch {epoch + 1}/{num_epochs}")
        start_time = time.time()

        # 训练单轮，获取训练损失
        epoch_loss = train_one_epoch(train_loader, encoder, encoder_optimizer, decoder, decoder_optimizer)
        loss_list.append(epoch_loss)

        # 计算训练耗时
        epoch_duration = time.time() - start_time
        # 打印训练损失和耗时
        print(f"Train MSE Loss: {epoch_loss:.4f}, Time: {epoch_duration:.2f} seconds")

        # 更新编码器学习率（按调度器配置）
        if encoder_scheduler:
            encoder_scheduler.step()
        # 更新解码器学习率（按调度器配置）
        if decoder_scheduler:
            decoder_scheduler.step()

        # 验证阶段
        start_time = time.time()
        val_loss = validate_one_epoch(val_loader, encoder, decoder)
        test_loss_list.append(val_loss)

        # 计算验证耗时
        epoch_duration = time.time() - start_time
        # 打印验证损失和耗时
        print(f"Test MSE Loss: {val_loss:.4f}, Time: {epoch_duration:.2f} seconds")

        # 保存最优模型（验证损失下降时更新）
        if val_loss < best_loss:
            best_loss = val_loss
            # 保存编码器权重（文档Phase1预训练输出：编码器表征模型）
            torch.save(encoder.state_dict(), save_path + '\Model\Phase1_encoder.pth')
            # 保存解码器权重（预留后续微调或分析使用）
            torch.save(decoder.state_dict(), save_path + '\Model\Phase1_decoder.pth')
            # 打印模型保存信息
            print(f"Model saved with validation loss: {val_loss:.4f}")


def load_model(model, save_path):
    """
    加载预训练模型权重（文档两阶段训练：Phase1预训练后，Phase2微调时加载此权重）
    输入：
        model - 待加载权重的模型实例
        save_path - 预训练权重文件路径
    输出：model - 加载权重后的模型实例
    """
    checkpoint = torch.load(save_path, weights_only=False)
    # 加载模型权重
    model.load_state_dict(checkpoint)
    return model


def Phase1():
    """
    文档核心：Phase1预训练（HCP数据集上的MAE预训练，对应论文两阶段训练的第一阶段）
    功能：训练STEncoder编码器，学习fMRI信号的通用语义表征，为Phase2微调做准备
    """
    # 序列段长列表（文档最优段长Nₛ=20，此处固定为20，与论文一致）
    length_list = [20]
    for len_idx in length_list:
        # 数据集路径（HCP数据集，文档Phase1预训练专用数据集）
        data_dir = r"C:\Users\20355\Desktop\CogReader-main\Data\HCP"
        # 结果保存路径（预训练模型、损失曲线等）
        save_path = r"C:\Users\20355\Desktop\CogReader-main\Results\Training\Phase1"
        # 模型保存子路径
        Model_path = save_path + "\Model"
        # 创建保存目录（不存在则创建，避免报错）
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(Model_path, exist_ok=True)
        # 预训练轮数（读取全局参数）
        num_epochs = EPOCH_NUM
        # 学习率（读取全局参数，与文档Phase1预训练LR=1e-4一致）
        learning_rate = LR

        # 加载模型超参数配置（Config类包含编码器/解码器的层数、维度等参数）
        cfg = Config()
        # 初始化时空编码器（文档：空间Transformer+时间Transformer，MAE编码器核心）
        encoder = STEncoder(cfg, len_idx)
        encoder = encoder.to("cuda")  # 转移到GPU
        # 初始化时空解码器（文档：MAE解码器，用于重构fMRI信号）
        decoder = STDecoder(cfg, len_idx)
        decoder = decoder.to("cuda")  # 转移到GPU

        # 初始化编码器优化器（Adam优化器，文档Phase1预训练指定优化器）
        encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
        # 初始化解码器优化器（Adam优化器，与编码器一致）
        decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

        # 打印数据加载提示
        print("loading data....")
        # 加载HCP数据集：划分训练集（80%）和验证集（20%），按段长len_idx=20分割序列
        train_data, val_data = create_dataloader(
            data_dir=data_dir,
            random_state=39,  # 随机种子（固定以保证可复现性）
            test_size=0.2,    # 验证集比例（文档实验配置）
            length=len_idx    # 序列段长（Nₛ=20，文档最优值）
        )

        # 创建训练数据加载器（batch_size=2，4个工作线程）
        train_loader = DataLoader(train_data, batch_size=2, num_workers=4)
        # 创建验证数据加载器（同训练配置）
        val_loader = DataLoader(val_data, batch_size=2, num_workers=4)

        # 打印数据加载成功信息
        print("loading data success!!!")
        print(f"train dataset: {len(train_loader)}")  # 打印训练批次数
        print(f"test dataset: {len(val_loader)}")    # 打印验证批次数
        print(f"Processing length: {len_idx}")       # 打印当前处理的序列段长

        # 初始化编码器学习率调度器（每5轮衰减为原来的0.5，文档实验配置）
        encoder_scheduler = StepLR(encoder_optimizer, step_size=5, gamma=0.5)
        # 初始化解码器学习率调度器（与编码器一致）
        decoder_scheduler = StepLR(decoder_optimizer, step_size=5, gamma=0.5)

        # 启动Phase1预训练（调用训练主函数）
        train(train_loader, val_loader, encoder, encoder_optimizer, encoder_scheduler,
              decoder, decoder_optimizer, decoder_scheduler,
              num_epochs, save_path)


if __name__ == '__main__':
    # 程序入口：启动Phase1预训练（文档两阶段训练的第一阶段核心）
    Phase1()