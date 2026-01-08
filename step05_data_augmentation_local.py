import json
import os
import time
from tqdm import tqdm
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- 1. 路径与设备自动适配 ---
# 自动检测是在 Kaggle 云端还是本地 PyCharm
if os.path.exists("/kaggle/input"):
    # 云端配置
    DATA_PATH = "/kaggle/input/nuaa-control-qa/control_knowledge_base"
    OUTPUT_PATH = "/kaggle/working/synthetic_data_5k.json"
    DEVICE = "cuda"
else:
    # 本地配置
    DATA_PATH = "./data/control_knowledge_base"
    OUTPUT_PATH = "./data/synthetic_data_5k.json"
    DEVICE = "cpu"  # 本地环境驱动未稳，强制使用 CPU 避免报错

print(f"🚀 当前运行环境: {'Kaggle云端' if DEVICE == 'cuda' else '本地电脑'}")
print(f"🧠 正在准备出题系统 (使用设备: {DEVICE})...")

# --- 2. 加载资源 ---
# 加载 Embedding 模型
embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")

# 加载知识库
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"找不到知识库路径: {DATA_PATH}，请检查文件夹是否存在。")

vector_db = FAISS.load_local(DATA_PATH, embeddings, allow_dangerous_deserialization=True)

# 提取所有原始文本片段
all_docs = [vector_db.docstore.search(vector_db.index_to_docstore_id[i]).page_content
            for i in range(len(vector_db.index_to_docstore_id))]

# 加载 Qwen 模型
model_dir = snapshot_download("Qwen/Qwen2.5-1.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map=DEVICE,
    torch_dtype="auto"
)


# --- 3. 核心生成逻辑 ---
def generate_qa_pair(context):
    # 强制要求 LaTeX 格式以适配 Obsidian [cite: 2025-10-24]
    prompt = f"""你现在是南航自动化学院的教授。请根据以下教材片段出三道题：
1. 一道单选题（含选项和答案）
2. 一道填空题（含答案）
3. 一道简答题（含答案）

要求：
- 所有的数学公式、变量（如 G(s), s, 传递函数等）必须使用 LaTeX 格式封装 [cite: 2025-10-24]。
- 复杂的公式请使用 $$...$$，行内变量使用 $...$ [cite: 2025-10-24]。
- 确保输出在 Obsidian 中能清晰明了地显示 [cite: 2025-12-17]。

教材片段：
{context}

请严格按 JSON 格式输出：
{{
  "choice_question": "",
  "choice_answer": "",
  "fill_question": "",
  "fill_answer": "",
  "short_question": "",
  "short_answer": ""
}}
"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)

    generated_ids = model.generate(**model_inputs, max_new_tokens=800)
    response = tokenizer.batch_decode(generated_ids[:, model_inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    return response


# --- 4. 循环生成 ---
generated_data = []
# 本地测试建议只跑 2 条，Kaggle 全量跑改为 len(all_docs)
test_limit = 2 if DEVICE == "cpu" else len(all_docs)

print(f"📝 开始出题任务，目标条数: {test_limit}")

for i in tqdm(range(test_limit)):
    try:
        raw_output = generate_qa_pair(all_docs[i])

        # --- 暴力提取逻辑：只拿第一个 { 和最后一个 } 之间的内容 ---
        start_idx = raw_output.find('{')
        end_idx = raw_output.rfind('}')
        if start_idx != -1 and end_idx != -1:
            clean_json = raw_output[start_idx:end_idx + 1]

            # 修复 LaTeX 导致的转义报错 [cite: 2025-10-24]
            clean_json = clean_json.replace("\\", "\\\\").replace("\\\\\\\\", "\\\\")

            generated_data.append(json.loads(clean_json))
        else:
            print(f"第 {i} 条未找到 JSON 结构，跳过")
    except Exception as e:
        print(f"跳过第 {i} 条记录，原因: {e}")
        continue

# --- 5. 保存结果 ---
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(generated_data, f, ensure_ascii=False, indent=2)

print(f"✅ 任务完成！结果已存至: {OUTPUT_PATH}")