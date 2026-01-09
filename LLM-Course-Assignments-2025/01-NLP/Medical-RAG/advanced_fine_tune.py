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
# ========================
# 路径配置
# ========================
DATASET_DIR = "/root/autodl-tmp/Medical-RAG/dataset"
TRAIN_FILE = os.path.join(DATASET_DIR, "alpaca_formatted_train_data_8k.json")
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
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True,use_cache=False)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    load_in_4bit=True,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
model = prepare_model_for_kbit_training(model)
# 数据加载与预处理
def load_and_process_dataset(path):
    df = pd.read_json(path, orient='records')# 读取 JSON 文件为 pandas DataFrame
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)# 读取 JSON 文件为 pandas DataFrame
    return Dataset.from_pandas(df)# 转为 Hugging Face Dataset 格式（便于后续 map 操作）



def preprocess_function(examples):
    instructions = examples['instruction']
    inputs = examples.get('input', [""] * len(instructions))
    outputs = examples['output']
    #通用 prompt
    full_texts = []
    for instr, inp, out in zip(instructions, inputs, outputs):
        text = (
            "<|im_start|>system\n你是一个有帮助的助手。<|im_end|>\n"
            f"<|im_start|>user\n{instr}\n{inp}<|im_end|>\n"
            f"<|im_start|>assistant\n{out}<|im_end|>"
        )
        full_texts.append(text)

    model_inputs = tokenizer(
        full_texts,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    labels = model_inputs["input_ids"].clone()
    for i in range(labels.size(0)):
        prompt = full_texts[i].split("<|im_start|>assistant\n")[0] + "<|im_start|>assistant\n"
        prompt_tokens = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        assistant_start = len(prompt_tokens)
        if assistant_start < labels.size(1):
            labels[i, :assistant_start] = -100

    model_inputs["labels"] = labels
    return model_inputs

train_dataset = load_and_process_dataset(TRAIN_FILE)
val_dataset = load_and_process_dataset(VAL_FILE)

print(f"  Train: {TRAIN_FILE}")
print(f"  Val:   {VAL_FILE} ({'存在' if has_validation else '不存在'})")
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
    
train_tokenized = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names
)
val_tokenized = val_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=val_dataset.column_names
)

print(f"✅ 预处理完成：训练集 {len(train_tokenized)} 条，验证集 {len(val_tokenized) if val_tokenized else 0} 条")

# ========================
# LoRA 配置
# ========================
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    bias="none",
    r=16,
    lora_alpha=64,
    lora_dropout=0.1,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

# 应用 LoRA
model = get_peft_model(model, peft_config)  # ← 此时 model 才是 PeftModel
model.print_trainable_parameters()

# 新增：检查是否有参数 requires_grad=True
trainable_params = [p for p in model.parameters() if p.requires_grad]
print(f"Number of trainable parameters: {len(trainable_params)}")
if len(trainable_params) == 0:
    raise RuntimeError("❌ 没有可训练参数！LoRA 未正确注入。")
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
    learning_rate=5e-4,
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