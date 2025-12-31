# advanced_fine_tune.py - 针对AutoDL的优化版本（已调整数据处理部分）
import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.optim.lr_scheduler import CosineAnnealingLR
import bitsandbytes as bnb  # 用于8-bit优化器

# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 加载数据集
dataset_path = "/root/autodl-tmp/Medical-RAG/dataset"
dataset = load_from_disk(dataset_path)

# 使用4-bit量化加载模型以节省显存
model = AutoModelForCausalLM.from_pretrained(
    "qwen/Qwen2.5-7B-Instruct",
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config={
        "load_in_4bit": True,
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.float16,
    },
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def format_medical_qa(example):
    """格式化医疗问答数据 - 根据实际数据集格式调整"""
    # 根据实际数据集格式处理
    if 'questions' in example and 'answers' in example:
        # 处理第一个问题和对应的答案
        questions = example['questions']
        answers = example['answers']
        
        # 从questions列表中取第一个问题
        if isinstance(questions, list) and len(questions) > 0:
            if isinstance(questions[0], list):
                # 如果questions[0]也是一个列表，取第一个问题
                question = questions[0][0] if len(questions[0]) > 0 else ""
            else:
                # 如果questions[0]是字符串
                question = questions[0]
        else:
            question = ""
        
        # 从answers列表中取第一个答案
        if isinstance(answers, list) and len(answers) > 0:
            answer = answers[0]
        else:
            answer = ""
    else:
        # 如果列名不匹配，返回第一个和第二个值
        values = list(example.values())
        question, answer = values[0], values[1] if len(values) > 1 else ""
    
    prompt = f"<|system|>\n请回答以下医疗相关问题：\n<|user|>\n{question}\n<|assistant|>\n{answer}"
    return prompt

def preprocess_dataset(dataset):
    def process_func(examples):
        texts = [format_medical_qa(ex) for ex in examples]
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=2048,
            return_attention_mask=False  # 使用Flash Attention时可设为False
        )
        
        labels = tokenized["input_ids"].copy()
        tokenized["labels"] = labels
        
        return tokenized
    
    # 检查数据集结构，打印列名以便调试
    print("数据集结构:")
    for split_name in dataset.keys():
        print(f"{split_name}: {dataset[split_name].column_names}")
    
    # 只处理训练集，如果验证集存在也处理
    train_dataset = dataset["train"]
    processed_train = train_dataset.map(
        process_func,
        batched=True,
        remove_columns=train_dataset.column_names,
        num_proc=4,  # 使用多进程加速
        desc="Tokenizing train dataset"
    )
    
    result_dataset = {"train": processed_train}
    
    if "validation" in dataset:
        val_dataset = dataset["validation"]
        processed_val = val_dataset.map(
            process_func,
            batched=True,
            remove_columns=val_dataset.column_names,
            num_proc=4,
            desc="Tokenizing validation dataset"
        )
        result_dataset["validation"] = processed_val
    
    return result_dataset

# 预处理数据集
processed_dataset = preprocess_dataset(dataset)

# LoRA配置 - 针对Qwen模型优化
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,  # 增加rank以提高性能
    lora_alpha=64,
    lora_dropout=0.1,
    target_modules=[
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

# 应用LoRA
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 训练参数配置 - 针对AutoDL优化
training_args = TrainingArguments(
    output_dir="./medical-qwen-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,  # 根据显存调整
    gradient_accumulation_steps=8,  # 调整以平衡速度和显存
    warmup_steps=500,
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="steps" if "validation" in processed_dataset else "no",
    eval_steps=500,
    learning_rate=2e-4,  # 稍微提高学习率
    fp16=True,
    logging_dir="./logs",
    save_total_limit=2,
    load_best_model_at_end=True if "validation" in processed_dataset else False,
    metric_for_best_model="eval_loss" if "validation" in processed_dataset else None,
    greater_is_better=False,
    dataloader_pin_memory=False,
    remove_unused_columns=False,  # 因为使用了自定义数据处理
    gradient_checkpointing=True,  # 激活梯度检查点以节省显存
    report_to=None,  # 禁用wandb等日志服务
)

# 数据整理器
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# 训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset['train'],
    eval_dataset=processed_dataset.get('validation'),
    data_collator=data_collator,
)

# 开始训练
print("开始微调训练...")
trainer.train()

# 保存最终模型
final_output_dir = "./medical-qwen-finetuned-final"
model.save_pretrained(final_output_dir)
tokenizer.save_pretrained(final_output_dir)
print(f"模型已保存到 {final_output_dir}")

# 合并LoRA权重到基础模型（可选）
print("正在合并LoRA权重...")
model_to_save = model.merge_and_unload()
model_to_save.save_pretrained("./medical-qwen-merged")
tokenizer.save_pretrained("./medical-qwen-merged")
print("合并后的模型已保存")