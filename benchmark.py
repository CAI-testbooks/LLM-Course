import time
import os
import torch
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- ⚙️ 配置区域 ---
# 如果是在本地跑，保持 "-1" (使用 CPU)
# 如果是在 Colab/Kaggle 跑，改成 "0" (使用 GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def run_benchmark():
    print("🚀 正在加载模型进行测速...")
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    model_dir = snapshot_download(model_name)

    # 自动判断设备：如果有 GPU 且 CUDA_VISIBLE_DEVICES 不是 -1，就用 GPU，否则 CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📊 当前测试设备: {device.upper()}")
    if device == "cpu":
        print("⚠️ 注意：CPU 速度通常较慢，这是正常的。")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map=device,
        torch_dtype="auto",
        trust_remote_code=True
    )

    # --- 🧪 开始测试 ---
    # 让它写长一点，测得才准
    prompt = "请详细介绍一下'自动控制原理'这门课程的主要内容，包括经典控制理论和现代控制理论的区别，字数在200字以上。"
    messages = [
        {"role": "system", "content": "你是一个助手。"},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    print("-" * 30)
    print("⏱️ 开始生成，请稍候...")

    # 1. 记录开始时间
    start_time = time.time()

    # 2. 生成内容 (强制生成至少 100 个 token 以确保测试有效)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,  # 允许生成的最大长度
        min_new_tokens=100  # 强制它多写点
    )

    # 3. 记录结束时间
    end_time = time.time()

    # --- 🧮 计算速度 ---
    # 提取新生成的 token (去掉输入的 token)
    input_token_len = model_inputs.input_ids.shape[1]
    output_token_len = generated_ids.shape[1]
    new_tokens = output_token_len - input_token_len

    duration = end_time - start_time
    speed = new_tokens / duration

    response = tokenizer.decode(generated_ids[0][input_token_len:], skip_special_tokens=True)

    print("-" * 30)
    print(f"✅ 生成完毕！")
    print(f"📝 生成内容预览: {response[:50]}...")  # 只打印前50个字看看
    print("-" * 30)
    print(f"🔢 生成 Token 数: {new_tokens}")
    print(f"⏱️ 耗时: {duration:.2f} 秒")
    print(f"🚀 推理速度 (Tokens/s): {speed:.2f}")
    print("-" * 30)


if __name__ == "__main__":
    run_benchmark()