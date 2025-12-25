# Multimodal Vision-Language Assistant for Autonomous Driving Scenarios
# 基于自动驾驶场景的多模态视觉语言助手

本项目旨在利用多模态大模型（Qwen2.5-VL-7B-Instruct）构建一个自动驾驶助手。通过对交通场景图像的理解，识别关键交通要素（车辆、行人、信号灯等），并解释其对驾驶行为的影响。项目包含数据处理、LoRA 微调以及模型推理的全套流程。

## 📂 目录结构

```
.
├── converted_images/          # 存放从数据集提取的图片
├── process_data.py            # 数据预处理脚本：读取 Parquet，保存图片，转换格式
├── process_imgtxt.py          # 辅助脚本：将数据转换为 Qwen2.5-VL 训练格式
├── train_lora_qwen25_vl.py    # LoRA 微调训练脚本
├── test_model.py              # 加载 LoRA 权重进行推理测试
├── qwen_test.py               # 基础模型（无 LoRA）推理测试
├── requirements.txt           # 项目依赖
├── qwen_finetune.json         # 训练数据 (示例)
└── README.md                  # 项目说明文档
```

## 🛠️ 环境准备

建议使用 Python 3.9+ 和支持 CUDA 的 GPU 环境。

1. **安装依赖**

```bash
pip install -r requirements.txt
```

主要依赖包括：
- `torch`, `transformers`, `peft`, `accelerate`
- `deepspeed` (用于训练加速)
- `flash-attn` (推荐安装以提高效率)

## 🚀 数据准备

本项目使用 Parquet 格式的数据集（CODA-LM）。

1. **下载模型权重 (Qwen2.5-VL-7B-Instruct)**

   ```bash
   # 配置 HF 镜像（确保下载速度） 
   export HF_ENDPOINT=https://hf-mirror.com 
   # 下载模型
   huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./Qwen2.5-VL-7B-Instruct --local-dir-use-symlinks False --resume-download # 断点续传，只补下损坏的文件
   ```

2. **运行数据处理脚本**

   该脚本会读取 Parquet 文件，将图片提取到 `converted_images/` 目录，并生成训练所需的 JSON 数据。

   ```bash
   python process_data.py
   ```

   *注意：请在 `process_data.py` 中修改 `parquet_path` 为你实际的数据路径。*

## 🏋️‍♂️ 模型微调 (LoRA Training)

使用 Qwen2.5-VL-7B-Instruct 模型进行 LoRA 微调。

1. **配置训练参数**

   打开 `train_lora_qwen25_vl.py`，根据你的环境修改以下配置：
   - `MODEL_NAME`: 基础模型路径 (e.g., `/root/autodl-fs/Qwen2.5-VL-7B-Instruct`)
   - `DATA_PATH`: 训练数据 JSON 路径
   - `OUTPUT_DIR`: 模型保存路径

2. **开始训练**

   ```bash
   python train_lora_qwen25_vl.py
   ```

   脚本会自动处理图像和文本的 Data Collation，并使用 PEFT 库进行高效微调。

## 🧪 推理与测试

### 1. 测试微调后的模型

使用 `test_model.py` 加载基础模型和训练好的 LoRA 权重进行推理。

```bash
python test_model.py
```
*需在脚本中指定 `BASE_MODEL_PATH`, `LORA_PATH` 和 `TEST_IMAGE_PATH`。*

### 2. 测试基础模型

使用 `qwen_test.py` 直接测试原始 Qwen2.5-VL 模型的效果，用于对比微调前后的性能。

```bash
python qwen_test.py
```

## 📝 任务描述

模型的主要任务是分析自车视角的交通图像，关注以下七类物体：
1. 车辆（轿车、卡车、公交车等）
2. 易受伤害的道路使用者（行人、骑行者）
3. 交通标志
4. 交通信号灯
5. 交通锥
6. 障碍物
7. 杂物

模型将描述这些物体的外观、位置、方向，并解释它们如何影响自车的驾驶决策。

