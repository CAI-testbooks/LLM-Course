# scripts/init_kb.py
from src.vector_store import VectorStoreManager
from src.document_processor import DocumentProcessor
from src.config import RAGConfig
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def main():
    config = RAGConfig.from_yaml("configs/config.yaml")

    # 处理文档
    processor = DocumentProcessor(config)
    chunks = processor.process_directory("./data/raw")

    # 创建向量数据库
    vector_store_manager = VectorStoreManager(config)
    vector_store = vector_store_manager.create_vector_store(chunks)

    print(f"知识库初始化完成，共处理 {len(chunks)} 个文本块")


if __name__ == "__main__":
    main()
