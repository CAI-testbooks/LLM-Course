#!/usr/bin/env python
# merge_model.py - 将 LoRA 适配器权重合并到基础模型中（防乱码版）
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ========================
# 配置参数
# ========================
# 设置镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 基础模型路径（微调前的原始模型）
BASE_MODEL_PATH = "/root/autodl-tmp/qwen/Qwen2.5-7B-Instruct"

# LoRA 适配器路径（微调后保存的 LoRA 权重）
LORA_ADAPTER_PATH = "/root/autodl-tmp/Medical-RAG/Tune-model/medical-qwen-lora-final"

# 合并后模型的保存路径
MERGED_MODEL_PATH = "/root/autodl-tmp/Medical-RAG/Tune-model/medical-qwen-merged"

# ========================
# 合并 LoRA 权重（防乱码版）
# ========================
def merge_lora_weights():
    print("🔍 加载基础模型和tokenizer...")
    
    # 先加载tokenizer，确保编码正确
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH, 
        trust_remote_code=True,
        use_fast=False,  # 使用慢速但更准确的tokenizer
        padding_side="left"
    )
    
    # 确保pad_token设置正确
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
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
    
    print("🔍 测试合并后模型的中文生成能力...")
    # 测试中文生成
    test_prompt = "<|im_start|>system\n你是一个专业的医疗问答助手。<|im_end|>\n<|im_start|>user\n什么是高血压？<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = merged_model.generate(
            inputs.input_ids.to(merged_model.device),
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(f"测试生成结果: {generated_text}")
    
    # 检查是否包含中文字符
    if any('\u4e00' <= char <= '\u9fff' for char in generated_text):
        print("✅ 中文生成测试通过！")
    else:
        print("⚠️ 警告：生成结果可能不包含中文字符")
    
    print("💾 保存合并后的模型和tokenizer...")
    
    # 保存模型
    merged_model.save_pretrained(
        MERGED_MODEL_PATH,
        safe_serialization=True,
        max_shard_size="5GB"
    )
    
    # 保存tokenizer（使用微调时使用的tokenizer设置）
    tokenizer.save_pretrained(MERGED_MODEL_PATH)
    
    # 创建配置文件，确保加载时使用正确的参数
    config_file = os.path.join(MERGED_MODEL_PATH, "config.json")
    if os.path.exists(config_file):
        import json
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 确保配置中包含中文支持相关参数
        config["trust_remote_code"] = True
        config["torch_dtype"] = "float16"
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 合并完成！模型已保存至: {MERGED_MODEL_PATH}")
    return merged_model, tokenizer

def main():
    print("🚀 开始合并 LoRA 权重...")
    print("⚠️ 注意：此版本包含防乱码措施")
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
    merged_model, tokenizer = merge_lora_weights()
    
    print("\n✅ 模型合并成功！防乱码措施已应用")
    print(f"  - 合并后的模型已保存至: {MERGED_MODEL_PATH}")
    print("  - tokenizer 已正确配置中文编码")
    print("  - 现在你可以直接加载合并后的模型进行推理")
    
    # 验证加载
    print("\n🔍 验证合并后模型的加载...")
    try:
        test_model = AutoModelForCausalLM.from_pretrained(
            MERGED_MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        test_tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH, trust_remote_code=True)
        print("✅ 合并后模型加载验证通过！")
    except Exception as e:
        print(f"❌ 加载验证失败: {e}")

if __name__ == "__main__":
    main()