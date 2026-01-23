# advanced_fine_tune.py - 针对 AutoDL 的优化微调脚本（Qwen2.5-7B + LoRA + 4-bit）
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import torch
import json
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
import matplotlib.pyplot as plt
import pandas as pd
from peft import prepare_model_for_kbit_training
from transformers import TrainerCallback

# ========================
# 路径配置
# ========================
DATASET_DIR = "/root/autodl-tmp/Medical-RAG/dataset"
TRAIN_FILE = os.path.join(DATASET_DIR, "alpaca_formatted_train_data.json")
VAL_FILE = os.path.join(DATASET_DIR, "alpaca_formatted_validation_data.json")
OUTPUT_BASE = "/root/autodl-tmp/Medical-RAG/Tune-model"
MODEL_PATH = "/root/autodl-tmp/qwen/Qwen2___5-7B-Instruct"

# 检查训练文件是否存在
if not os.path.exists(TRAIN_FILE):
    raise FileNotFoundError(f"训练数据文件不存在: {TRAIN_FILE}")

has_validation = os.path.exists(VAL_FILE)
print("✅ 数据文件检查完成")

# ========================
# 模型与 Tokenizer
# ========================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, 
    trust_remote_code=True,
    use_cache=False,
    padding_side="right"  # 关键：右padding避免影响生成停止
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # 确保生成时从左到右，停止信号有效

# 4-bit量化配置优化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,  # 使用显式的量化配置
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False  # 训练时关闭cache，避免干扰
model.config.pretraining_tp = 1

# ========================
# 数据加载与预处理（核心优化：解决标签错误导致的重复）
# ========================
def load_and_process_dataset(path):
    df = pd.read_json(path, orient='records')
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.strip()  # 去除首尾空格，避免无效token
    # 过滤空数据
    df = df[(df['instruction'].notna()) & (df['output'].notna())]
    return Dataset.from_pandas(df)

def preprocess_function(examples):
    instructions = examples['instruction']
    inputs = examples.get('input', [""] * len(instructions))
    outputs = examples['output']
    full_texts = []
    
    # 优化prompt模板：明确结束标识，引导模型停止生成
    for instr, inp, out in zip(instructions, inputs, outputs):
        # 关键修改：在assistant回复末尾加入明确的结束标记
        text = (
            "<|im_start|>system\n你是一个有帮助的助手。回答要求：1. 条理清晰；2. 禁止重复表述；3. 回答时，不做冗余推理。<|im_end|>\n"
            f"<|im_start|>user\n{instr}\n{inp}<|im_end|>\n"
            f"<|im_start|>assistant\n{out}<|im_end|>"  # 保留原始结束标记，强化停止信号
        )
        full_texts.append(text)

    # 优化tokenizer参数：避免截断assistant部分
    model_inputs = tokenizer(
        full_texts,
        max_length=256,  
        truncation=True,
        truncation_strategy="only_first",  # 优先截断prompt，保留回答
        padding="max_length",
        return_tensors="pt",
        return_attention_mask=True  # 显式返回attention mask
    )

    # 核心优化：精准计算assistant部分起始位置，避免标签错误
    labels = model_inputs["input_ids"].clone()
    labels[:] = -100  # 先全部置为-100（不计算损失）
    
    for i in range(len(full_texts)):
        # 拆分prompt和回答部分
        prompt_part = full_texts[i].split("<|im_start|>assistant\n")[0] + "<|im_start|>assistant\n"
        # 计算prompt部分的token数（不添加额外special token，避免偏移）
        prompt_tokens = tokenizer(
            prompt_part, 
            add_special_tokens=False,  # 关键：和full_texts的tokenization保持一致
            return_attention_mask=False
        )["input_ids"]
        assistant_start_idx = len(prompt_tokens)
        
        # 确保索引不越界
        if assistant_start_idx < model_inputs["input_ids"].shape[1]:
            # 只对assistant部分计算损失
            labels[i, assistant_start_idx:] = model_inputs["input_ids"][i, assistant_start_idx:]
            
            # 额外优化：将回答末尾的<|im_end|>也计入损失，强化停止信号
            end_token = tokenizer("<|im_end|>", add_special_tokens=False)["input_ids"][0]
            labels[i, assistant_start_idx:] = torch.where(
                model_inputs["input_ids"][i, assistant_start_idx:] == end_token,
                end_token,
                labels[i, assistant_start_idx:]
            )

    model_inputs["labels"] = labels
    model_inputs["attention_mask"] = model_inputs["attention_mask"].bool()  # 确保mask类型正确
    return model_inputs

train_dataset = load_and_process_dataset(TRAIN_FILE)
val_dataset = load_and_process_dataset(VAL_FILE) if has_validation else None

print(f"  Train: {TRAIN_FILE} (有效数据：{len(train_dataset)})")
print(f"  Val:   {VAL_FILE} ({'存在' if has_validation else '不存在'}，有效数据：{len(val_dataset) if val_dataset else 0})")

if len(train_dataset) > 0:
    print("\n🔍 训练集示例:")
    print(train_dataset[0])
else:
    raise ValueError("训练集为空，请检查数据格式或过滤条件")

# 预处理数据集
train_tokenized = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names,
    #num_proc=os.cpu_count()  # 多进程加速
)

val_tokenized = None
if val_dataset and len(val_dataset) > 0:
    val_tokenized = val_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=val_dataset.column_names,
        #num_proc=os.cpu_count()
    )

print(f"✅ 预处理完成：训练集 {len(train_tokenized)} 条，验证集 {len(val_tokenized) if val_tokenized else 0} 条")

# ========================
# LoRA 配置（微调参数降低过拟合）
# ========================
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    bias="none",
    r=8,  # 降低r值，减少过拟合风险
    lora_alpha=32,  # 对应r值调整
    lora_dropout=0.15,  # 增大dropout，抑制过拟合
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    use_rslora=True,  # 提升LoRA稳定性
)

# 应用 LoRA
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 检查可训练参数
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.4f}%)")
if trainable_params == 0:
    raise RuntimeError("❌ 没有可训练参数！LoRA 未正确注入。")

# ========================
# 训练参数（核心优化：防止过拟合+提升稳定性）
# ========================
training_args = TrainingArguments(
    output_dir=os.path.join(OUTPUT_BASE, "medical-qwen-lora"),
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_ratio=0.05,  # 改用比例，适配不同数据集大小
    logging_steps=10,
    save_steps=300,
    evaluation_strategy="steps" if val_tokenized else "no",
    eval_steps=300,
    learning_rate=2e-4,  # 降低学习率，减少过拟合（原5e-4偏大）
    fp16=True,
    fp16_full_eval=True,  # 验证时也用fp16，提升效率和稳定性
    logging_dir=os.path.join(OUTPUT_BASE, "logs"),
    save_total_limit=2,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    gradient_checkpointing=True,
    load_best_model_at_end=True if val_tokenized else False,
    metric_for_best_model="eval_loss" if val_tokenized else None,
    greater_is_better=False,
    gradient_checkpointing_kwargs={"use_reentrant": False},  # 适配新版PyTorch
    report_to=["tensorboard"],
    optim="adamw_torch",
    weight_decay=0.01,  # 加入权重衰减，抑制过拟合
    max_grad_norm=1.0,  # 梯度裁剪，防止梯度爆炸导致训练不稳定
    lr_scheduler_type="cosine",  # 余弦学习率衰减，让训练更平稳
)

# ========================
# Data Collator
# ========================
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8,  # 对齐硬件，提升效率
    return_tensors="pt"
)

# ========================
# 回调函数
# ========================
train_losses = []
eval_losses = []
steps = []

class LossLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            train_losses.append(logs["loss"])
            steps.append(state.global_step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and "eval_loss" in metrics:
            eval_losses.append(metrics["eval_loss"])

callbacks = [LossLoggingCallback()]

# ========================
# 启动训练
# ========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    data_collator=data_collator,
    callbacks=callbacks
)

print("\n🚀 开始LORA微调训练...")
trainer.train()

# ========================
# 保存模型
# ========================
final_lora_dir = os.path.join(OUTPUT_BASE, "medical-qwen-lora-final")
model.save_pretrained(final_lora_dir)
tokenizer.save_pretrained(final_lora_dir)
print(f"\n✅ LoRA 适配器已保存至: {final_lora_dir}")

# ========================
# 可视化训练loss展示
# ========================
if len(train_losses) > 0:
    plt.figure(figsize=(12, 5))

    # train—loss
    plt.subplot(1, 2, 1)
    plt.plot(steps, train_losses, label="Train Loss", marker='o', markersize=3)
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # eval-loss
    if eval_losses:
        eval_steps_list = [training_args.eval_steps * (i+1) for i in range(len(eval_losses))]
        plt.subplot(1, 2, 2)
        plt.plot(eval_steps_list, eval_losses, label="Eval Loss", color="red", marker='s', markersize=3)
        plt.title("Evaluation Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    loss_save_path = os.path.join(OUTPUT_BASE, "logs", "loss_save.png")
    plt.savefig(loss_save_path, dpi=150)
    plt.close()
    print(f"✅ Loss 曲线已保存至: {loss_save_path}")


print("🎉LORA 微调完成！")