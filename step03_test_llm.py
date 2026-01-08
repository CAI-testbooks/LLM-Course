import os

# --- 关键修改 1: 屏蔽显卡，强行使用 CPU ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def load_and_chat():
    print("🚀 正在准备下载/加载 Qwen2.5-1.5B 模型...")

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    # 1. 下载模型
    model_dir = snapshot_download(model_name)

    # 2. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    # 3. 加载模型 (强制使用 CPU)
    print("⚠️ 检测到 RTX 50系显卡兼容性问题，正在切换至 CPU 模式运行...")
    print("🧠 正在加载模型 (这可能需要十几秒)...")

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="cpu",  # --- 关键修改 2: 指定跑在 CPU 上 ---
        torch_dtype="auto",  # CPU 自动选择精度
        trust_remote_code=True
    )

    print("✅ 模型加载成功！准备对话...")

    # --- 测试对话 ---
    prompt = "你是谁？请用一句话介绍一下什么是'自动控制'。"
    messages = [
        {"role": "system", "content": "你是一个来自南京航空航天大学(NUAA)航天学院的专业课程助教，你的名字叫'南航小智'。请用专业、亲切的语气回答学生的问题。"},
        {"role": "user", "content": prompt}
    ]

    # 处理输入
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # --- 关键修改 3: 输入数据也要放在 CPU 上 ---
    model_inputs = tokenizer([text], return_tensors="pt").to("cpu")

    # 生成回答
    print(f"\nUser: {prompt}")
    print("AI (Qwen) 正在思考... (CPU 可能会慢一点点，请耐心等待)\n")

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )

    # 解码输出
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"Qwen: {response}")


if __name__ == "__main__":
    load_and_chat()