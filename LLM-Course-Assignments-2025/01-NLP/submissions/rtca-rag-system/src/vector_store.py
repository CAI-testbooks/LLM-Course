# src/vector_store.py
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from .config import RAGConfig


class VectorStoreManager:
    """向量数据库管理器"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            model_kwargs={'device': config.device},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store = None

    def create_vector_store(self, chunks: List[Dict]) -> FAISS:
        """创建向量数据库"""
        texts = [chunk['content'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]

        self.vector_store = FAISS.from_texts(
            texts=texts,
            embedding=self.embedding_model,
            metadatas=metadatas
        )

        # 保存向量数据库
        self.vector_store.save_local(self.config.vector_db_path)
        return self.vector_store

    def load_vector_store(self) -> FAISS:
        """加载向量数据库"""
        self.vector_store = FAISS.load_local(
            self.config.vector_db_path,
            self.embedding_model,
            allow_dangerous_deserialization=True
        )
        return self.vector_store

    def update_vector_store(self, new_chunks: List[Dict]):
        """更新向量数据库"""
        if self.vector_store is None:
            self.load_vector_store()

        texts = [chunk['content'] for chunk in new_chunks]
        metadatas = [chunk['metadata'] for chunk in new_chunks]

        self.vector_store.add_texts(texts, metadatas)
        self.vector_store.save_local(self.config.vector_db_path)
