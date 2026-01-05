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
# ========================
# 路径配置
# ========================
DATASET_DIR = "/root/autodl-tmp/Medical-RAG/dataset"
TRAIN_FILE = os.path.join(DATASET_DIR, "train_data_8k.json")
VAL_FILE = os.path.join(DATASET_DIR, "validation_data.json")
OUTPUT_BASE = "/root/autodl-tmp/Medical-RAG/Tune-model"

# 检查训练文件是否存在
if not os.path.exists(TRAIN_FILE):
    raise FileNotFoundError(f"训练数据文件不存在: {TRAIN_FILE}")

has_validation = os.path.exists(VAL_FILE)
print("✅ 数据文件检查完成")
print(f"  Train: {TRAIN_FILE}")
print(f"  Val:   {VAL_FILE} ({'存在' if has_validation else '不存在'})")

# ========================
# 修正的数据集加载函数（处理JSONL格式）
# ========================
def load_jsonl_like(file_path):
    """正确加载JSONL格式的数据集"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 尝试直接解析JSON数组
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    
    # 如果是真正的JSONL格式（每行一个JSON对象）
    try:
        records = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    except json.JSONDecodeError:
        pass
    
    # 如果是格式错误的JSON数组（缺少逗号或引号错误）
    # 按 }{ 分割进行修复
    records = []
    for obj_str in content.strip().split('}{'):
        if not obj_str.startswith('{'):
            obj_str = '{' + obj_str
        if not obj_str.endswith('}'):
            obj_str = obj_str + '}'
        try:
            records.append(json.loads(obj_str))
        except json.JSONDecodeError:
            continue  # 跳过损坏行
    return records

def create_flat_dataset(file_path):
    """创建扁平化的数据集"""
    raw_records = load_jsonl_like(file_path)
    
    # 展平为 {"question": "...", "answer": "..."} 列表
    flat_data = []
    for idx, rec in enumerate(raw_records):
        if isinstance(rec, dict):
            questions_list = rec.get("questions", [])
            answers_list = rec.get("answers", [])
            
            # 处理多个问题对应一个答案的情况
            if answers_list:
                answer = str(answers_list[0]) if answers_list else ""
                for q_list in questions_list:
                    if isinstance(q_list, list):
                        for q in q_list:  # 支持多问一答
                            flat_data.append({
                                "question": str(q).strip(), 
                                "answer": answer.strip()
                            })
                    elif isinstance(q_list, str):
                        # 如果问题字段直接是字符串
                        flat_data.append({
                            "question": q_list.strip(), 
                            "answer": answer.strip()
                        })
    
    return Dataset.from_list(flat_data)

# 加载数据集
train_dataset = create_flat_dataset(TRAIN_FILE)
val_dataset = create_flat_dataset(VAL_FILE) if has_validation else None

print("\n🔍 训练集示例:")
if len(train_dataset) > 0:
    print(train_dataset[0])
else:
    print("训练集为空，请检查数据格式")

if val_dataset and len(val_dataset) > 0:
    print("\n🔍 验证集示例:")
    print(val_dataset[0])

# 检查数据结构
print("\n📊 数据集结构信息:")
print(f"训练集列名: {train_dataset.column_names}")
print(f"训练集大小: {len(train_dataset)}")
if val_dataset:
    print(f"验证集大小: {len(val_dataset)}")

# ========================
# 模型与 Tokenizer
# ========================
MODEL_PATH = "/root/autodl-tmp/qwen/Qwen2___5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    load_in_4bit=True,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
model.enable_input_require_grads()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

# ========================
# 构建 Qwen 对话模板
# ========================
def format_qwen_prompt(question: str, answer: str) -> str:
    return f"""<|im_start|>system
你是一个专业的医疗问答助手。<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>"""

def preprocess_function(examples):
    """处理扁平化数据的预处理函数，掩码非assistant部分"""
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []
    
    # 获取样本数量
    num_samples = len(examples["question"]) if "question" in examples else 0
    
    # 遍历每个样本
    for i in range(num_samples):
        question = str(examples["question"][i]).strip()
        answer = str(examples["answer"][i]).strip()
        
        # 跳过空数据
        if not question or not answer:
            continue
            
        # 构建完整的prompt
        full_prompt = format_qwen_prompt(question, answer)
        
        # 构建仅包含system和user部分的prompt（用于确定掩码位置）
        user_prompt = f"""<|im_start|>system
你是一个专业的医疗问答助手。<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
        
        # 分词处理
        full_tokens = tokenizer(
            full_prompt,
            truncation=True,
            max_length=1024,
            padding=False,
            return_tensors="pt",
        )
        
        user_tokens = tokenizer(
            user_prompt,
            truncation=True,
            max_length=1024,
            padding=False,
            return_tensors="pt",
        )
        
        # 获取input_ids
        input_ids = full_tokens["input_ids"][0]
        
        # 创建labels，初始化为-100（忽略loss计算）
        labels = torch.full_like(input_ids, -100)
        
        # 找到assistant部分开始的位置
        user_len = len(user_tokens["input_ids"][0])
        
        # 确保assistant部分在序列范围内
        if user_len < len(input_ids):
            # 从assistant部分开始的位置设置labels为实际token
            labels[user_len:] = input_ids[user_len:]
        
        # 创建attention_mask
        attention_mask = torch.ones_like(input_ids)
        
        batch_input_ids.append(input_ids)
        batch_attention_mask.append(attention_mask)
        batch_labels.append(labels)

    # 如果没有有效数据，返回空字典
    if len(batch_input_ids) == 0:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    # 填充到相同长度
    max_length = max(len(ids) for ids in batch_input_ids)
    
    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []
    
    for input_ids, attention_mask, labels in zip(batch_input_ids, batch_attention_mask, batch_labels):
        # 填充input_ids
        if len(input_ids) < max_length:
            pad_len = max_length - len(input_ids)
            padded_input_ids.append(torch.cat([input_ids, torch.full((pad_len,), tokenizer.pad_token_id)]))
            padded_attention_mask.append(torch.cat([attention_mask, torch.zeros((pad_len,))]))
            padded_labels.append(torch.cat([labels, torch.full((pad_len,), -100)]))
        else:
            padded_input_ids.append(input_ids)
            padded_attention_mask.append(attention_mask)
            padded_labels.append(labels)
    
    return {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_mask),
        "labels": torch.stack(padded_labels)
    }

# ========================
# 预处理数据集
# ========================
print("\n🔄 正在预处理数据...")
train_tokenized = train_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=train_dataset.column_names,
    desc="Tokenizing train set"
).filter(lambda x: len(x["input_ids"]) > 0)

if val_dataset:
    val_tokenized = val_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing validation set"
    ).filter(lambda x: len(x["input_ids"]) > 0)
else:
    val_tokenized = None

print(f"✅ 预处理完成：训练集 {len(train_tokenized)} 条，验证集 {len(val_tokenized) if val_tokenized else 0} 条")

# ========================
# LoRA 配置
# ========================
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=64,
    lora_dropout=0.1,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    init_lora_weights=False,
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ========================
# 训练参数
# ========================
training_args = TrainingArguments(
    output_dir=os.path.join(OUTPUT_BASE, "medical-qwen-lora"),
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_steps=300,
    logging_steps=50,
    save_steps=300,
    evaluation_strategy="steps" if val_tokenized else "no",
    eval_steps=300,#每 300 步评估一次（可选）
    learning_rate=2e-4,
    fp16=True,
    logging_dir=os.path.join(OUTPUT_BASE, "logs"),
    save_total_limit=2,
    load_best_model_at_end=True if val_tokenized else False,
    metric_for_best_model="eval_loss" if val_tokenized else None,
    greater_is_better=False,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    gradient_checkpointing=True,
    report_to=["tensorboard"],

)

# ========================
# Data Collator
# ========================
# 
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 测试 collator
sample_batch = [train_tokenized[i] for i in range(min(2, len(train_tokenized)))]
try:
    batch = data_collator(sample_batch)
    print("✅ Data collator 测试通过！")
    print("Batch keys:", list(batch.keys()))
    print("input_ids shape:", batch["input_ids"].shape)
except Exception as e:
    print("❌ Data collator 报错:", e)
    raise
# ========================
# 可视化训练loss
# ========================
from transformers import TrainerCallback

train_losses = []
eval_losses = []
steps = []
# 回调函数
class LossLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            train_losses.append(logs["loss"])
            steps.append(state.global_step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and "eval_loss" in metrics:
            eval_losses.append(metrics["eval_loss"])

loss_callback = LossLoggingCallback()



# ========================
# 启动训练
# ========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    data_collator=data_collator,
    callbacks=[loss_callback],  # ←←← 新增回调，进行loss可视化操作
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
        # eval 每 eval_steps 一次，从 eval_steps 开始
        eval_steps_list = [i * training_args.eval_steps for i in range(1, len(eval_losses) + 1)]
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
    plt.close()  # 避免在 notebook 中显示
    print(f"✅ Loss 曲线已保存至: {loss_save_path}")

print("🎉LORA 微调完成！")