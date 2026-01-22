import json
import os
import sys
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 强制刷新输出，防止看不到打印信息
def v_print(msg):
    print(msg, flush=True)


# 1. 路径设置
data_folder = r"D:\workerspace\control_qa\data"
json_file = os.path.join(data_folder, "final_control_data_5050.json")
textbook_file = os.path.join(data_folder, "textbook.txt")
db_save_path = r"D:\workerspace\control_qa\vector_db"

documents = []

# --- 2. 处理 JSON 问答对 ---
v_print("🔍 步骤 1: 开始处理 JSON 问答对...")
if not os.path.exists(json_file):
    v_print(f"❌ 错误: 找不到 JSON 文件: {json_file}")
else:
    with open(json_file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            v_print(f"📖 成功读取 JSON，包含 {len(data)} 条原始数据")
            for item in data:
                q = item.get('question', '')
                a = item.get('answer', '')
                if q and a:
                    content = f"问题: {q}\n回答: {a}"
                    doc = Document(page_content=content, metadata={"source": "精选问答库"})
                    documents.append(doc)
        except Exception as e:
            v_print(f"❌ JSON 解析失败: {e}")

# --- 3. 处理教材文本 ---
v_print("🔍 步骤 2: 开始处理教材原文...")
if os.path.exists(textbook_file):
    text = ""
    try:
        with open(textbook_file, 'r', encoding='utf-8') as f:
            text = f.read()
    except:
        with open(textbook_file, 'r', encoding='gbk', errors='ignore') as f:
            text = f.read()

    if text:
        v_print(f"📖 成功读取教材，长度: {len(text)} 字符")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            doc = Document(page_content=chunk, metadata={"source": f"教材原文-第{i}段"})
            documents.append(doc)
    else:
        v_print("⚠️ 警告: 教材内容为空！")

# --- 4. 向量化与保存 ---
v_print(f"📊 步骤 3: 检查装载情况... 当前共有 {len(documents)} 个知识片段")

if len(documents) == 0:
    v_print("❌ 严重错误: 没有加载到任何有效数据，停止构建数据库！")
    sys.exit()

v_print("🚀 正在加载 Embedding 模型 (首次运行可能较慢)...")
# 使用 RTX 5070 的算力加速
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cuda'}
)

v_print("💎 正在生成向量索引并计算相似度矩阵...")
vector_db = FAISS.from_documents(documents, embeddings)

v_print(f"💾 正在保存数据库到: {db_save_path}")
vector_db.save_local(db_save_path)

v_print("🎉 🎉 🎉 恭喜！数据库构建成功！")