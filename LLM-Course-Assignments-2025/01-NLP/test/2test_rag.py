from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. 加载同样的嵌入模型
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 2. 加载已经持久化的向量数据库
vectordb = Chroma(persist_directory="./chroma_db_medical", embedding_function=embedding_model)


def test_retrieval(query):
    print(f"\n用户提问: {query}")
    print("-" * 30)

    # 执行相似度搜索，提取前 3 个相关片段
    results = vectordb.similarity_search_with_score(query, k=3)

    for doc, score in results:
        # score 越小越相似（Chroma 默认是 L2 距离，有些版本是余弦距离需确认，但通常 score < 1.0 表示相关）
        print(f"【相关度 Score】: {score:.4f}")
        print(f"【来源】: {doc.metadata.get('source')} - {doc.metadata.get('original_question')}")
        print(f"【内容片段】: ...{doc.page_content[:100]}...")  # 只打印前100字
        print("\n")


if __name__ == "__main__":
    # 测试几个医疗问题
    test_retrieval("感冒了嗓子疼怎么办？")
    test_retrieval("糖尿病患者饮食需要注意什么？")