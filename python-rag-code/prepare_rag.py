"""
准备RAG系统的嵌入部分
"""
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import json


class RAGEmbeddingPreparer:
    def __init__(self, model_path="all-MiniLM-L6-v2"):
        print("初始化RAG嵌入准备器...")

        # 尝试加载本地模型，如果没有则下载
        local_path = f"./models/{model_path}"
        if os.path.exists(local_path):
            print(f"从本地加载模型: {local_path}")
            self.model = SentenceTransformer(local_path)
        else:
            print(f"下载模型: {model_path}")
            self.model = SentenceTransformer(f'sentence-transformers/{model_path}')
            # 保存到本地
            os.makedirs("./models", exist_ok=True)
            self.model.save(local_path)
            print(f"模型已保存到: {local_path}")

        print(f"模型维度: {self.model.get_sentence_embedding_dimension()}")

    def create_sample_python_docs(self):
        """创建示例Python文档数据"""
        print("\n创建示例Python文档...")

        python_docs = [
            {
                "id": 1,
                "content": "Python中的open()函数用于打开文件。基本语法: open(filename, mode)。模式包括'r'读取、'w'写入、'a'追加等。",
                "metadata": {"topic": "file_io", "language": "python"}
            },
            {
                "id": 2,
                "content": "使用with语句可以自动管理文件资源，确保文件正确关闭。示例: with open('file.txt', 'r') as f: content = f.read()",
                "metadata": {"topic": "file_io", "language": "python"}
            },
            {
                "id": 3,
                "content": "列表推导式是Python创建列表的简洁方式。语法: [expression for item in iterable if condition]。示例: squares = [x**2 for x in range(10)]",
                "metadata": {"topic": "list_comprehension", "language": "python"}
            },
            {
                "id": 4,
                "content": "装饰器是修改函数或类的行为的高级功能。使用@符号应用装饰器。def decorator(func): def wrapper(): print('Before'); func(); print('After'); return wrapper",
                "metadata": {"topic": "decorators", "language": "python"}
            },
            {
                "id": 5,
                "content": "Virtual environments allow you to manage project-specific dependencies. Create: python -m venv myenv. Activate: source myenv/bin/activate (Linux) or myenv\\Scripts\\activate (Windows)",
                "metadata": {"topic": "virtual_env", "language": "python"}
            }
        ]

        # 保存文档
        os.makedirs("./data", exist_ok=True)
        with open("./data/python_docs.json", "w", encoding="utf-8") as f:
            json.dump(python_docs, f, ensure_ascii=False, indent=2)

        print(f" 创建了 {len(python_docs)} 个示例文档")
        return python_docs

    def generate_and_save_embeddings(self, documents):
        """生成并保存嵌入向量"""
        print("\n生成文档嵌入向量...")

        # 提取文档内容
        texts = [doc["content"] for doc in documents]
        doc_ids = [doc["id"] for doc in documents]

        # 分批生成嵌入（避免内存问题）
        batch_size = 8
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            print(f"  处理批次 {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")

            batch_embeddings = self.model.encode(
                batch_texts,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            all_embeddings.append(batch_embeddings)

        # 合并
        embeddings = np.vstack(all_embeddings)

        # 保存嵌入向量
        embedding_data = {
            "embeddings": embeddings,
            "doc_ids": doc_ids,
            "dimension": embeddings.shape[1],
            "model": "all-MiniLM-L6-v2"
        }

        with open("./data/document_embeddings.pkl", "wb") as f:
            pickle.dump(embedding_data, f)


        print(f"  文档数量: {len(documents)}")
        print(f"  嵌入维度: {embeddings.shape[1]}")
        print(f"  文件位置: ./data/document_embeddings.pkl")

        return embeddings

    def test_semantic_search(self):
        """测试语义搜索"""
        print("\n=== 语义搜索测试 ===")

        # 加载保存的数据
        with open("./data/python_docs.json", "r", encoding="utf-8") as f:
            documents = json.load(f)

        with open("./data/document_embeddings.pkl", "rb") as f:
            data = pickle.load(f)
            embeddings = data["embeddings"]

        # 测试查询
        queries = [
            "How to read files in Python?",
            "什么是列表推导式？",
            "How to create virtual environment?"
        ]

        for query in queries:
            print(f"\n查询: '{query}'")

            # 生成查询嵌入
            query_embedding = self.model.encode([query], normalize_embeddings=True)[0]

            # 计算相似度
            similarities = np.dot(embeddings, query_embedding)

            # 获取最相似的3个文档
            top_k = 3
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            print(f"最相关的 {top_k} 个文档:")
            for i, idx in enumerate(top_indices):
                doc = documents[idx]
                print(f"  {i + 1}. [相似度: {similarities[idx]:.4f}] {doc['content'][:80]}...")


# 主程序
if __name__ == "__main__":


    # 1. 初始化
    preparer = RAGEmbeddingPreparer()

    # 2. 创建示例数据
    documents = preparer.create_sample_python_docs()

    # 3. 生成嵌入向量
    embeddings = preparer.generate_and_save_embeddings(documents)

    # 4. 测试搜索
    preparer.test_semantic_search()

