import json
import os
import re
import time
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- 1. 环境与路径配置 ---
# 必须先在 Kaggle 右侧 Add Input 挂载这两个数据集 [cite: 2025-12-24]
DATA_PATH = "/kaggle/input/nuaa-control-qa/control_knowledge_base"
# MODEL_PATH 需指向你挂载的模型权重数据集中的具体目录 [cite: 2025-12-24]
MODEL_PATH = "/kaggle/input/qwen25-15b-weights/qwen_files/Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_PATH = "/kaggle/working/synthetic_data_5k.json"
DEVICE = "cuda"

print(f"🚀 生产环境就绪。模型路径: {MODEL_PATH}")

# --- 2. 加载本地资源 (0秒预热模式) ---
print("🧠 正在从挂载硬盘加载知识库与模型...")
# 使用本地 Embedding 模型 [cite: 2025-11-12]
embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
vector_db = FAISS.load_local(DATA_PATH, embeddings, allow_dangerous_deserialization=True)

# 提取所有原始文本片段 [cite: 2025-12-24]
all_docs = [vector_db.docstore.search(vector_db.index_to_docstore_id[i]).page_content
            for i in range(len(vector_db.index_to_docstore_id))]

# 离线加载 Qwen2.5-1.5B [cite: 2025-12-24]
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype="auto"
)


# --- 3. 5倍产出出题函数 (达标核心) ---
def generate_qa_pair(context):
    # 强制要求 LaTeX 格式以适配 Obsidian [cite: 2025-10-24, 2025-12-17]
    prompt = f"""你现在是南航自动化学院教授。请根据教材片段产出 5 道不同类型的考题以扩充题库：
1. 单选题 A (含选项及答案)
2. 单选题 B (含选项及答案)
3. 填空题 (含答案)
4. 简答题 (含答案)
5. 计算分析题 (需包含复杂的 LaTeX 推导过程及答案)

要求：
- 所有的数学公式（如 $G(s)$, $\omega_n$, 传递函数等）必须使用 LaTeX 格式封装 [cite: 2025-10-24]。
- 复杂的公式请使用 $$...$$，行内变量使用 $...$ [cite: 2025-12-17]。
- 确保输出在 Obsidian 中清晰明了 [cite: 2025-12-17]。

教材片段：
{context}

请严格按 JSON 格式输出，不要有任何解释内容：
{{
  "choice_1": {{"q": "", "options": ["A.","B.","C.","D."], "a": ""}},
  "choice_2": {{"q": "", "options": ["A.","B.","C.","D."], "a": ""}},
  "fill": {{"q": "", "a": ""}},
  "short": {{"q": "", "a": ""}},
  "calc": {{"q": "", "a": ""}}
}}
"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)

    # 适当增加 tokens 限制以容纳 5 道题的内容 [cite: 2025-12-24]
    generated_ids = model.generate(**model_inputs, max_new_tokens=1536)
    response = tokenizer.batch_decode(generated_ids[:, model_inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    return response


# --- 4. 鲁棒性生成循环 ---
generated_data = []
test_limit = len(all_docs)

print(f"🔥 5000+ 目标计划启动！原始素材共: {test_limit} 条")

for i in tqdm(range(test_limit)):
    try:
        raw_output = generate_qa_pair(all_docs[i])

        # A. 使用正则暴力提取 JSON 块，解决多余文字干扰 [cite: 2025-12-24]
        json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if not json_match:
            continue
        clean_json = json_match.group()

        # B. 核心修复：针对 LaTeX 反斜杠进行 JSON 转义处理 [cite: 2025-12-24]
        # 逻辑：将单反斜杠替换为双反斜杠，防止 json.loads 崩溃 [cite: 2025-12-24]
        clean_json = clean_json.replace('\\', '\\\\').replace('\\\\\\\\', '\\\\')

        # C. 解析并存储
        data_item = json.loads(clean_json)
        generated_data.append(data_item)

        # D. 实时日志监控：解决 Kaggle UI 刷新延迟问题 [cite: 2025-12-24]
        if (i + 1) % 10 == 0:
            total_est = len(generated_data) * 5
            print(
                f"📊 进度: {i + 1}/{test_limit} | 估算已存题量: {total_est} | 成功率: {(len(generated_data) / (i + 1)) * 100:.1f}%")

    except Exception:
        continue

# --- 5. 最终保存 ---
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(generated_data, f, ensure_ascii=False, indent=2)

print(f"✅ 全部任务完成！")
print(f"📊 最终有效题目总数: {len(generated_data) * 5}")  # 每条数据含 5 题 [cite: 2025-12-24]
print(f"📂 结果文件: {OUTPUT_PATH}")