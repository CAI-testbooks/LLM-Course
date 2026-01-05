#!/usr/bin/env python
# merge_model.py - 将 LoRA 适配器权重合并到基础模型中
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ========================
# 配置参数
# ========================
# 设置镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 基础模型路径（微调前的原始模型）
BASE_MODEL_PATH = "/root/autodl-tmp/qwen/Qwen2.5-7B-Instruct"

# LoRA 适配器路径（微调后保存的 LoRA 权重）
LORA_ADAPTER_PATH = "/root/autodl-tmp/Medical-RAG/Tune-model/medical-qwen-lora-final"

# 合并后模型的保存路径
MERGED_MODEL_PATH = "/root/autodl-tmp/Medical-RAG/Tune-model/medical-qwen-merged"

# ========================
# 合并 LoRA 权重
# ========================
def merge_lora_weights():
    print("🔍 加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    print("🔍 加载 LoRA 适配器...")
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    
    print("🔧 合并 LoRA 权重到基础模型...")
    merged_model = model.merge_and_unload()
    
    print("💾 保存合并后的模型...")
    merged_model.save_pretrained(
        MERGED_MODEL_PATH,
        safe_serialization=True,  # 使用安全序列化保存
        max_shard_size="5GB"      # 分片保存，避免单个文件过大
    )
    
    # 保存 tokenizer（通常与基础模型相同）
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    tokenizer.save_pretrained(MERGED_MODEL_PATH)
    
    print(f"✅ 合并完成！模型已保存至: {MERGED_MODEL_PATH}")
    return merged_model

def main():
    print("🚀 开始合并 LoRA 权重...")
    print(f"  基础模型路径: {BASE_MODEL_PATH}")
    print(f"  LoRA 适配器路径: {LORA_ADAPTER_PATH}")
    print(f"  合并后保存路径: {MERGED_MODEL_PATH}")
    
    # 检查必要文件是否存在
    if not os.path.exists(BASE_MODEL_PATH):
        raise FileNotFoundError(f"基础模型路径不存在: {BASE_MODEL_PATH}")
    
    if not os.path.exists(LORA_ADAPTER_PATH):
        raise FileNotFoundError(f"LoRA 适配器路径不存在: {LORA_ADAPTER_PATH}")
    
    # 创建输出目录
    os.makedirs(MERGED_MODEL_PATH, exist_ok=True)
    
    # 执行合并
    merged_model = merge_lora_weights()
    
    print("\n✅ 模型合并成功！")
    print(f"  - 合并后的模型已保存至: {MERGED_MODEL_PATH}")
    print("  - 现在你可以直接加载合并后的模型进行推理")

if __name__ == "__main__":
    main()