#!/usr/bin/env python3
"""
构建向量数据库
"""
import os
import json
import time
import shutil

print("=" * 60)
print("构建向量数据库")
print("=" * 60)

# 1. 清理旧数据库
if os.path.exists("./chroma_db"):
    print("清理旧的向量数据库...")
    try:
        shutil.rmtree("./chroma_db")
        print(" 清理完成")
    except Exception as e:
        print(f"清理警告: {e}")

# 2. 加载数据
data_path = "./data/document_chunks.json"
if not os.path.exists(data_path):
    print(f" 数据文件不存在: {data_path}")
    exit(1)

print(f"加载数据文件...")
try:
    with open(data_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    print(f" 加载成功: {len(documents)} 个文本块")
except Exception as e:
    print(f" 加载失败: {e}")
    exit(1)

# 3. 导入所需库
print("\n加载嵌入模型...")
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
except ImportError as e:
    print(f" 缺少库: {e}")
    print("请运行: pip install langchain-community chromadb")
    exit(1)



#  正确配置：不要在encode_kwargs中添加show_progress_bar
embeddings = HuggingFaceEmbeddings(
    model_name="./models/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={
        'normalize_embeddings': True,
        'batch_size': 32
    }
)


# 5. 构建向量数据库
print("\n" + "-" * 60)
print(f"开始构建向量数据库 ({len(documents)}个文档)...")
start_time = time.time()

try:
    # Chroma.from_texts会自动显示进度条
    vector_db = Chroma.from_texts(
        texts=documents,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_name="python_docs"
    )

    # 持久化
    vector_db.persist()

    elapsed = time.time() - start_time


    # 验证
    if os.path.exists("./chroma_db"):
        size = sum(os.path.getsize(os.path.join("./chroma_db", f))
                   for f in os.listdir("./chroma_db")
                   if os.path.isfile(os.path.join("./chroma_db", f)))
        print(f"   数据库大小: {size / 1024 / 1024:.1f} MB")



except Exception as e:
    print(f"\n 构建失败: {e}")

    # 尝试更简单的备选方案

    try:
        # 创建一个临时的向量存储测试
        test_docs = documents[:100]  # 先测试100个
        test_db = Chroma.from_texts(
            texts=test_docs,
            embedding=embeddings,
            persist_directory="./chroma_db_test"
        )
        print(f" 测试成功！可以处理 {len(test_docs)} 个文档")

    except Exception as e2:
        print(f" 测试也失败: {e2}")