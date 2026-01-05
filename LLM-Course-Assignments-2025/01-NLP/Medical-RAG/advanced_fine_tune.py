# advanced_fine_tune.py - 针对 AutoDL 的优化微调脚本（Qwen2.5-7B + LoRA + 4-bit）
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import torch
from datasets import load_dataset
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
# 加载数据集（JSONL格式）
# ========================
def load_jsonl_dataset(file_path):
    """加载JSONL格式的数据集"""
    return load_dataset("json", data_files=file_path, split="train")

train_dataset = load_jsonl_dataset(TRAIN_FILE)
val_dataset = load_jsonl_dataset(VAL_FILE) if has_validation else None

print("\n🔍 训练集示例:")
print(train_dataset[0])
if val_dataset:
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
    """处理JSONL格式数据的预处理函数"""
    batch_prompts = []
    
    # 遍历每个样本
    for i in range(len(examples.get("questions", []))):
        # 获取问题列表
        questions_list = examples["questions"][i] if i < len(examples["questions"]) else []
        # 获取答案
        answer = examples["answers"][i] if i < len(examples["answers"]) else ""
        
        # 处理问题列表
        if isinstance(questions_list, list) and len(questions_list) > 0:
            # 使用第一个问题作为主要问题
            question = str(questions_list[0]).strip()
        else:
            question = ""
        
        # 处理答案
        answer = str(answer).strip()
        
        # 跳过空数据
        if not question or not answer:
            continue
            
        # 构建prompt
        prompt = format_qwen_prompt(question, answer)
        batch_prompts.append(prompt)

    # 如果没有有效数据，返回空字典
    if len(batch_prompts) == 0:
        return {"input_ids": [], "attention_mask": []}

    # 分词处理
    tokenized = tokenizer(
        batch_prompts,
        truncation=True,
        max_length=1024,
        padding="max_length",
        return_tensors="pt",
    )
    return tokenized

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