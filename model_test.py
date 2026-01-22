import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# 1. 修正后的路径 (使用 r 前缀防止转义)
base_model_path = r"D:\workerspace\models\Qwen\Qwen2___5-1___5B-Instruct"
lora_path = r"D:\workerspace\control_qa\model\qwen-lora"

# 【验证步骤】先检查路径是否存在，防止模型加载到一半报错
if not os.path.exists(base_model_path):
    print(f"❌ 错误：找不到基础模型路径，请检查：{base_model_path}")
elif not os.path.exists(lora_path):
    print(f"❌ 错误：找不到 LoRA 权重路径，请检查：{lora_path}")
else:
    print("✅ 路径检查通过，准备加载模型至 RTX 5070...")

    # 2. 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    # 使用 dtype=torch.float16 减少显存占用
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    # 3. 加载微调补丁 (LoRA)
    model = PeftModel.from_pretrained(model, lora_path)
    print("🚀 模型与微调权重加载完成！")

    # 4. 测试提问
    prompt = "什么是闭环控制系统的稳定性？"
    messages = [
        {"role": "system", "content": "你是一个自动控制原理专家。"},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7
    )

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("\n--- 专家回答 ---")
    print(response.split("assistant\n")[-1])