import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, Dataset, DatasetDict
import os
import matplotlib.pyplot as plt
import pandas as pd
from transformers import EarlyStoppingCallback

# ==================== 配置区 ====================
MODEL_PATH = "/root/model/Qwen/Qwen3-0___6B"
DATASET_PATH = "./data_extract/chinese_history_qa.json"
OUTPUT_DIR = "/root/autodl-tmp/qwen_history_lora"
MAX_SEQ_LENGTH = 1024

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

# LoRA 参数
R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# 训练超参数
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3
WARMUP_STEPS = 100
LOGGING_STEPS = 10  # 每 10 步记录一次
SAVE_STEPS = 200
EVAL_STEPS = 100  # 新增：每 100 步评估一次（可选）

# 新增：用于记录指标的列表
train_losses = []
eval_losses = []  # 如果你想加验证集的话
steps = []

# ================================================

# 1-2. 加载 tokenizer、模型、LoRA 配置（保持不变）
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",  # 开启 atten-fast 加速
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.gradient_checkpointing_enable()

# 3-4. 数据加载和预处理
df = pd.read_json(DATASET_PATH, orient='records')
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype(str)
dataset = Dataset.from_pandas(df)

dataset = dataset.train_test_split(test_size=0.1)
datasets = DatasetDict({
    "train": dataset["train"],
    "validation": dataset["test"],
})

def preprocess_function(examples):
    instructions = examples['instruction']
    inputs = examples.get('input', [""] * len(instructions))
    outputs = examples['output']

    full_texts = []
    for instr, inp, out in zip(instructions, inputs, outputs):
        text = (
            "<|im_start|>system\n你是一个党史研究专家。回答时必须严格基于真实历史事实，不允许任何杜撰、推测或虚构内容。如果问题超出你的知识范围或不确定，请直接回复“不知道”或“我无法确认具体细节”。党史知识具有严肃性和准确性要求，请确保：1. 回答要基于事实，准确引用历史事件的时间、地点和人物。2. 对于重要历史事件和决策，要体现其历史背景和意义。3. 如果上下文信息不足，请明确说明并建议查阅权威党史资料。4. 回答要体现党史教育的严肃性和教育意义<|im_end|>\n"
            f"<|im_start|>user\n请基于真实历史知识回答以下问题：\n{instr}\n{inp}<|im_end|>\n"
            f"<|im_start|>assistant\n{out}<|im_end|>\n"
        )
        full_texts.append(text)

    # 关键：使用 padding="max_length" 或动态，但这里我们用 tokenizer 统一处理
    # 推荐：直接让 tokenizer 返回 tensor，并设置 padding 到 max_length（或动态）
    model_inputs = tokenizer(
        full_texts,
        max_length=MAX_SEQ_LENGTH,
        truncation=True,
        padding="max_length",  # 先用固定长度 padding，避免 collator 出错
        return_tensors="pt",  # 直接返回 tensor
    )
    labels = model_inputs["input_ids"].clone()
    for i in range(labels.size(0)):
        # 重建当前样本的提示部分（不含 output）
        prompt = full_texts[i].split("<|im_start|>assistant\n")[0] + "<|im_start|>assistant\n"
        prompt_tokens = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        assistant_start = len(prompt_tokens)

        if assistant_start < labels.size(1):
            labels[i, :assistant_start] = -100

    model_inputs["labels"] = labels
    return model_inputs


tokenized_dataset = datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=["instruction", "input", "output"]
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ==================== 新增：自定义回调记录 loss ====================
from transformers import TrainerCallback


class LossLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            train_losses.append(logs["loss"])
            steps.append(state.global_step)
            print(f"Step {state.global_step} - Loss: {logs['loss']:.4f}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and "eval_loss" in metrics:
            eval_losses.append(metrics["eval_loss"])
            print(f"Step {state.global_step} - Eval Loss: {metrics['eval_loss']:.4f}")


# 实例化回调
loss_callback = LossLoggingCallback()
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=3,      # 验证集指标连续3次没有改善就停止
    early_stopping_threshold=0.01,  # 改善幅度小于0.01%视为没有改善
)

# ==================== 训练参数（新增 eval_strategy） ====================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    warmup_steps=WARMUP_STEPS,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    eval_steps=EVAL_STEPS,  # 新增：评估频率
    eval_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,  # 新增：加载最佳模型
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=False,
    bf16=True,
    optim="paged_adamw_8bit",
    # optim="adamw_torch",
    report_to="none",
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    gradient_checkpointing=True,
)

# ==================== Trainer（添加 callbacks） ====================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    callbacks=[loss_callback,early_stopping_callback],  # 新增：添加回调
)

# ==================== 开始训练 ====================
print("开始训练...")
trainer.train()

# ==================== 保存 LoRA ====================
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"LoRA 权重已保存到 {OUTPUT_DIR}")

# ==================== 绘制 loss 曲线 ====================
if len(train_losses) > 0:
    plt.figure(figsize=(12, 5))

    # 训练 loss
    plt.subplot(1, 2, 1)
    plt.plot(steps, train_losses, label="Train Loss", marker='o')
    plt.title("Training Loss Curve")
    plt.xlabel("Global Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # 验证 loss（如果有）
    if eval_losses:
        eval_steps = list(range(EVAL_STEPS, len(steps) * LOGGING_STEPS, EVAL_STEPS))
        plt.subplot(1, 2, 2)
        plt.plot(eval_steps[:len(eval_losses)], eval_losses, label="Eval Loss", color="orange", marker='s')
        plt.title("Evaluation Loss")
        plt.xlabel("Global Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
    plt.show()
    print(f"Loss 曲线已保存到 {OUTPUT_DIR}/loss_curve.png")