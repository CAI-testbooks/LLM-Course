import os
# 必须放在 import datasets 之前！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import tqdm

# ================= 配置项 =================
# 1. 嵌入模型：推荐使用 BAAI/bge-small-zh-v1.5，中文效果极佳且轻量
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
# 2. 向量库保存路径
PERSIST_DIRECTORY = "./chroma_db_medical"
# 3. 数据集名称 (使用 Lite 版本或从主版本切片)
# DATASET_NAME = "FreedomIntelligence/Huatuo-26M"
DATASET_NAME = "FreedomIntelligence/Huatuo26M-Lite"
# 4. 截取数据量 (作业要求5k，我们取6k以防清洗掉一部分)
SAMPLE_SIZE = 6000


def load_and_process_data():
    print(f"正在加载数据集: {DATASET_NAME}...")
    # stream=True 允许我们不下载整个26M数据集，只流式获取需要的部分
    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)

    docs = []
    print(f"正在处理前 {SAMPLE_SIZE} 条数据...")

    # 迭代获取数据
    iterator = iter(dataset)
    for i in tqdm.tqdm(range(SAMPLE_SIZE)):
        try:
            item = next(iterator)

            # Huatuo 数据通常包含 'question' 和 'answer' (具体字段名需根据实际下载的子集确认，通常是 instruction/output)
            # 这里假设字段为 'instruction' (问题) 和 'output' (回答)
            question = item.get("instruction", "") or item.get("question", "")
            answer = item.get("output", "") or item.get("answer", "")

            if not question or not answer:
                continue

            # --- 核心步骤：构建适合 RAG 的文本块 ---
            # 策略：将 Q 和 A 拼在一起。
            # 这样检索时，用户的 Query 既能匹配到原来的 Question，也能匹配到 Answer 中的关键词。
            page_content = f"问题：{question}\n答案：{answer}"

            # --- 核心步骤：添加元数据用于引用 ---
            # 作业要求：提供可解释的引用来源。
            # 我们把问题的来源（如果有）或者 ID 存入 metadata。
            metadata = {
                "source": "Huatuo-26M",
                "original_question": question[:50] + "..."  # 用于展示引用时的标题
            }

            docs.append(Document(page_content=page_content, metadata=metadata))

        except StopIteration:
            break

    print(f"原始文档处理完成，共 {len(docs)} 条。")
    return docs


def split_documents(docs):
    print("正在进行文本分块 (Chunking)...")
    # 对于医疗数据，段落可能较长，使用递归字符分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,  # 每个块的大小（字符数）
        chunk_overlap=64,  # 重叠部分，防止切断关键词
        separators=["\n\n", "\n", "。", "！", "？"]  # 优先按段落和句子切分
    )

    splitted_docs = text_splitter.split_documents(docs)
    print(f"分块完成，共生成 {len(splitted_docs)} 个向量块。")
    return splitted_docs


def create_vector_db(splitted_docs):
    print(f"正在加载嵌入模型 {EMBEDDING_MODEL_NAME} (第一次运行可能需要下载)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print("正在构建并保存 Chroma 向量数据库...")
    # 批量处理，防止内存溢出
    batch_size = 1000
    for i in range(0, len(splitted_docs), batch_size):
        batch = splitted_docs[i:i + batch_size]
        if i == 0:
            vectordb = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=PERSIST_DIRECTORY
            )
        else:
            vectordb.add_documents(batch)

    print(f"向量数据库构建完成！已保存至 {PERSIST_DIRECTORY}")


if __name__ == "__main__":
    # 1. 加载与清洗
    raw_docs = load_and_process_data()
    # 2. 分块
    chunks = split_documents(raw_docs)
    # 3. 向量化存储
    create_vector_db(chunks)