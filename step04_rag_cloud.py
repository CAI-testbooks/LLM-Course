import os
import time
import torch
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- 1. 路径适配 (关键修改) ---
# 在 Kaggle 运行时，使用云端路径；在本地备份时，使用本地路径
if os.path.exists("/kaggle/input"):
    DATA_PATH = "/kaggle/input/nuaa-control-qa/control_knowledge_base"
    DEVICE = "cuda"  # 云端使用 GPU
else:
    # 这里的路径改为你本地电脑的实际路径
    DATA_PATH = "./data/control_knowledge_base"
    DEVICE = "cpu"  # 本地因为驱动问题暂时用 CPU

print(f"🧠 正在加载模型与向量库 (运行设备: {DEVICE})...")

# --- 2. 核心逻辑 ---
# 加载 Embedding 模型
embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")

# 加载向量库
# allow_dangerous_deserialization=True 是因为本地读取自己生成的 pkl 文件是安全的
vector_db = FAISS.load_local(DATA_PATH, embeddings, allow_dangerous_deserialization=True)

# 加载 Qwen2.5-1.5B 模型
model_dir = snapshot_download("Qwen/Qwen2.5-1.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# 根据设备加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map=DEVICE,
    torch_dtype="auto"
)


def rag_chat(query):
    # 检索
    docs = vector_db.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    # 构造 Prompt
    prompt = f"你是一个南航航天学院的助教。请根据以下参考资料回答问题。\n参考资料：\n{context}\n问题：{query}"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)

    # 生成回答
    generated_ids = model.generate(**model_inputs, max_new_tokens=512)
    response = tokenizer.batch_decode(generated_ids[:, model_inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    return response


if __name__ == "__main__":
    question = "什么是自动控制系统的稳态误差？"
    print(f"问题: {question}")
    # 注意：本地 CPU 跑这段会比较慢，大约需要 1 分钟
    answer = rag_chat(question)
    print(f"回答: {answer}")