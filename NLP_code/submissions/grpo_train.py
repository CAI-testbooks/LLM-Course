import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
import json
import gc

# 加载模型和分词器
model_path = "/root/autodl-tmp/Meta-Llama-3-8B-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    dtype=torch.bfloat16,
    device_map="auto",
    torch_dtype=torch.bfloat16,  # 显式指定数据类型
    low_cpu_mem_usage=True,      # 减少CPU内存使用
    offload_folder="offload",    # 设置offload文件夹
    load_in_4bit=True,           # 使用4位量化
)

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

# 启用梯度检查点
model.gradient_checkpointing_enable()

# 修正的奖励函数
def correctness_reward_func(prompts, completions, answers=None, **kwargs):
    if answers is None:
        answers = kwargs.get('answers', None)
        if answers is None:
            return [0.0] * len(prompts)
    
    rewards = []
    for completion, answer in zip(completions, answers):
        model_output = completion[0]['content'].strip()
        true_answer = answer
        rewards.append(2.0 if model_output == true_answer else 0.0)
    
    # 手动清理内存
    del completions
    gc.collect()
    torch.cuda.empty_cache()
    
    return rewards

# 配置训练参数 - 减小所有尺寸参数
training_args = GRPOConfig(
    output_dir="/root/autodl-tmp/GRPO",
    learning_rate=1e-5,  # 降低学习率
    per_device_train_batch_size=1,  # 减小批处理大小
    gradient_accumulation_steps=8,  # 增加梯度累积步数
    max_prompt_length=256,  # 减小最大提示长度
    max_completion_length=48,  # 减小最大完成长度
    num_generations=2,  # 减少生成样本数量
    optim="adamw_8bit",
    num_train_epochs=1,
    bf16=True,
    remove_unused_columns=False,
    logging_steps=1,
    report_to=None,
    gradient_checkpointing=True,  # 启用梯度检查点
    fp16=False,  # 禁用fp16
    bf16_full_eval=True,  # 使用bf16进行评估
)

# 加载自定义数据集
def load_custom_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 添加 answers 字段到每个样本
    for sample in data:
        sample['answers'] = sample['answer']
    
    # 只加载部分数据用于测试
    return data[:50]  # 仅使用前50个样本

dataset = load_custom_dataset("/root/autodl-tmp/train_converted.json")

# 初始化 GRPO 训练器
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[correctness_reward_func],
    args=training_args,
    train_dataset=dataset,
)

# 开始训练
try:
    trainer.train()
    trainer.save_model("/root/autodl-tmp/GRPO")
except Exception as e:
    print(f"训练失败: {e}")
    # 尝试保存部分结果
    trainer.save_model("/root/autodl-tmp/GRPO_partial")