import os
import shutil
from datasets import load_dataset 
from tqdm import tqdm
from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# --- 1. 配置参数 ---
LOCAL_DATASET_PATH = "./Huatuo26M-Lite" 

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "medical_rag"

# 设置为 None，表示处理所有数据
MAX_DOCS = None  
BATCH_SIZE = 100 

# 本地模型路径
LOCAL_MODEL_PATH = "./BAAI/bge-m3"

# --- 2. 初始化模型 ---
print(f"正在加载本地 Embedding 模型: {LOCAL_MODEL_PATH}...")
embed_model = HuggingFaceEmbedding(
    model_name=LOCAL_MODEL_PATH, 
    device="cuda", 
    trust_remote_code=True
)
Settings.embedding_model = embed_model

# --- 3. 准备向量数据库 ---
print(f"正在初始化 ChromaDB: {CHROMA_PATH}")
# 清理旧数据，避免重复堆叠
if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH) 

db = chromadb.PersistentClient(path=CHROMA_PATH)
chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

def stream_documents():
    print(f"📂 正在从本地文件夹加载数据集: {LOCAL_DATASET_PATH}...")
    
    if not os.path.exists(LOCAL_DATASET_PATH):
        print(f"❌ 错误：找不到路径 {LOCAL_DATASET_PATH}")
        print("请先在终端运行 git clone 命令下载数据集！")
        return

    try:
        dataset = load_dataset(LOCAL_DATASET_PATH, split="train")
        print(f"📊 数据集加载成功，共 {len(dataset)} 条数据")
    except Exception as e:
        print(f"加载失败: {e}")
        return

    current_batch = []
    count = 0

    # 遍历整个数据集
    for i, item in tqdm(enumerate(dataset), desc="Processing", total=len(dataset)):
        if not item.get('question') or not item.get('answer'):
            continue

        text_chunk = f"问题：{item['question']}\n\n回答：{item['answer']}"
        
        doc = Document(
            text=text_chunk,
            metadata={
                "source": "Huatuo-26M-Lite",
                "original_question": item['question']
            },
            excluded_llm_metadata_keys=['source', 'original_question']
        )
        
        current_batch.append(doc)
        
        # 批处理
        if len(current_batch) >= BATCH_SIZE:
            yield current_batch
            current_batch = []
            
        count += 1
        if MAX_DOCS is not None and count >= MAX_DOCS:
            break
    
    # 处理剩余的文档
    if current_batch:
        yield current_batch

# --- 4. 执行构建 ---
def build():
    # 显式传递 embed_model
    index = VectorStoreIndex.from_vector_store(
        vector_store, 
        storage_context=storage_context,
        embed_model=embed_model
    )
    
    total_chunks = 0
    for batch_docs in stream_documents():
        # 使用 insert_nodes 批量插入
        index.insert_nodes(batch_docs)
        total_chunks += len(batch_docs)
        # 打印进度
        if total_chunks % 1000 == 0:
            print(f" --> 已入库 {total_chunks} 条数据")

    print(f"\n✅ 全量构建完成！共计 {total_chunks} 个向量块已存入 {CHROMA_PATH}")

if __name__ == "__main__":
    build()