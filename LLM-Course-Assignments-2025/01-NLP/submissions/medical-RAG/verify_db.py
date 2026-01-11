import os
import chromadb
import torch
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext

# --- 配置参数---
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "medical_rag"
LOCAL_MODEL_PATH = "./BAAI/bge-m3"

def check_chroma_direct():
    print("-" * 50)
    print("🔍 [阶段一] 底层数据检查 (Direct Inspection)")
    print("-" * 50)
    
    if not os.path.exists(CHROMA_PATH):
        print(f"❌ 错误：找不到目录 {CHROMA_PATH}，请先运行构建脚本！")
        return False

    try:
        # 连接数据库
        db = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = db.get_collection(COLLECTION_NAME)
        
        # 1. 检查数据量
        count = collection.count()
        print(f"📊 数据库当前存储条目数: {count}")
        
        if count == 0:
            print("⚠️ 警告：数据库是空的！构建过程可能出错。")
            return False
            
        # 2. 抽查第一条数据 (Peek)
        # 检查是否包含 text 和 metadata
        data = collection.peek(limit=1)
        if data and data['documents']:
            print(f"\n📝 [数据抽样]:")
            print(f"   ID: {data['ids'][0]}")
            print(f"   Metadatas: {data['metadatas'][0]}")
            print(f"   Text (前100字): {data['documents'][0][:100]}...")
        else:
            print("⚠️ 警告：无法读取数据内容。")
            
        print("✅ 底层数据结构正常。")
        return True
        
    except Exception as e:
        print(f"❌ Chroma 读取失败: {e}")
        return False

def check_semantic_retrieval():
    print("\n" + "-" * 50)
    print("🧠 [阶段二] 语义检索测试 (Semantic Test)")
    print("-" * 50)

    print(f"⏳ 正在加载本地模型 {LOCAL_MODEL_PATH} (用于将测试问题向量化)...")
    try:
        embed_model = HuggingFaceEmbedding(
            model_name=LOCAL_MODEL_PATH,
            device="cuda" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True
        )
        
        # 连接 LlamaIndex
        db = chromadb.PersistentClient(path=CHROMA_PATH)
        chroma_collection = db.get_collection(COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # 加载索引
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=embed_model
        )
        
        # 创建检索器
        retriever = index.as_retriever(similarity_top_k=3)
        
        # 测试问题 (针对 Huatuo 数据的常见问题)
        test_query = "感冒了头痛怎么办？"
        print(f"\n❓ 测试提问: '{test_query}'")
        
        results = retriever.retrieve(test_query)
        
        if not results:
            print("❌ 检索失败：未返回任何结果。")
            return

        print(f"🎉 检索成功！找到了 {len(results)} 条相关结果：\n")
        
        for i, node in enumerate(results):
            print(f"--- [结果 {i+1}] (相似度得分: {node.score:.4f}) ---")
            # 打印内容预览
            content_preview = node.node.get_content().replace('\n', ' ')[:150]
            print(f"📄 内容: {content_preview}...")
            # 打印元数据
            print(f"🔗 来源: {node.metadata.get('source', 'Unknown')}")
            print("")

    except Exception as e:
        print(f"❌ 语义检索测试失败: {e}")

if __name__ == "__main__":
    if check_chroma_direct():
        check_semantic_retrieval()