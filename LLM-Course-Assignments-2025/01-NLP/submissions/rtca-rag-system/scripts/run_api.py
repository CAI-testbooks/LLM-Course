# scripts/run_api.py
from src.api_app import FastAPIApp
from src.rag_system import RAGSystem
from src.retriever import HybridRetriever
from src.vector_store import VectorStoreManager
from src.config import RAGConfig
import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def main():
    config = RAGConfig.from_yaml("configs/config.yaml")

    # 加载向量数据库和文档块
    vector_store_manager = VectorStoreManager(config)
    vector_store = vector_store_manager.load_vector_store()

    # 加载文档块
    with open(os.path.join(config.paths.knowledge_base, "processed_chunks.json"), 'r') as f:
        chunks = json.load(f)

    # 创建检索器
    retriever = HybridRetriever(vector_store, chunks, config)

    # 创建RAG系统
    rag_system = RAGSystem(config)
    rag_system.retriever = retriever

    # 启动FastAPI应用
    api_app = FastAPIApp(rag_system)
    api_app.run(host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
