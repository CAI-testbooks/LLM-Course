import os
import sys

# 导入必要的库
# 教学点：这里展示了 LangChain 的模块化设计 (加载器、切割器、向量库、模型)
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ==========================================
# 配置区域 (这里是学生唯一需要修改的地方)
# ==========================================

# 1. 设置 API Key (建议使用 硅基流动 或 DeepSeek)
# 请将下面的 'sk-xxxxxxxx' 替换为你申请到的真实 Key
os.environ["OPENAI_API_KEY"] = "sk-gyuofotkkugmqvlmcuchjdzmipktruzczqvqtqyiyfqbqvsu"

# 2. 设置 Base URL (指向国内中转服务，无需 VPN)
os.environ["OPENAI_API_BASE"] = "https://api.siliconflow.cn/v1"

# 3. 定义模型名称
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # 大语言模型
EMBEDDING_MODEL = "BAAI/bge-m3"  # 向量模型


# ==========================================
# 主程序逻辑
# ==========================================

def main():
    print("-" * 50)
    print("🏥 中文医疗领域智能问答系统启动...")
    print("-" * 50)

    # ---------------------------------------------------
    # 第一步：数据加载 (Load)
    # 教学讲解：将非结构化的文本文件加载到内存中
    # ---------------------------------------------------
    print(f"[1/5] 正在加载医疗知识库 knowledge.txt ...")
    try:
        loader = TextLoader("./knowledge.txt", encoding="utf-8")
        docs = loader.load()
    except FileNotFoundError:
        print("❌ 错误：找不到 knowledge.txt 文件，请先生成医疗知识库数据文件！")
        return

    # ---------------------------------------------------
    # 第二步：文本切分 (Split)
    # 教学讲解：为了适应 LLM 的上下文窗口，我们需要把长文本切成小块
    # ---------------------------------------------------
    print(f"[2/5] 正在切分文本...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # 每个块的大小
        chunk_overlap=50  # 块之间的重叠部分，防止语义中断
    )
    splits = text_splitter.split_documents(docs)
    print(f"      >>> 文档已切分为 {len(splits)} 个片段")

    # ---------------------------------------------------
    # 第三步：向量化与存储 (Embed & Store)
    # 教学讲解：将文本片段转化为向量，并存入 ChromaDB 数据库
    # ---------------------------------------------------
    print(f"[3/5] 正在构建向量索引 (这可能需要几秒钟)...")

    # 定义向量模型 (Embedding)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # 存入向量库
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )
    print("      >>> 向量数据库构建完成！")

    # ---------------------------------------------------
    # 第四步：构建 RAG 链 (Chain)
    # 教学讲解：将 检索(Retriever) + 提示词(Prompt) + 模型(LLM) 串联起来
    # ---------------------------------------------------
    print(f"[4/5] 正在初始化大模型 ({MODEL_NAME})...")

    # 1. 定义检索器 (只查找最相关的 3 条信息)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 2. 定义大语言模型
    llm = ChatOpenAI(
        model_name=MODEL_NAME,
        temperature=0.1  # 温度越低，回答越严谨
    )

    # 3. 定义 Prompt 模板
    template = """
    你是一个专业的医疗AI助手。请严格根据以下【医学知识】来回答用户的问题。
    如果上下文中没有答案，请直接说"根据现有医学资料，我无法提供确切答案，建议咨询专业医生"，不要编造信息。

    【医学知识】：
    {context}

    【用户问题】：
    {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 4. 组装 RAG 链 (LangChain LCEL 语法)
    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    # ---------------------------------------------------
    # 第五步：演示提问 (Invoke)
    # 教学讲解：实际运行效果展示
    # ---------------------------------------------------
    print("-" * 50)

    # 测试问题 1
    query1 = "高血压的常见症状和治疗方法有哪些？"
    print(f"👤 提问: {query1}")
    print("🤖 思考中...", end="", flush=True)
    response1 = rag_chain.invoke(query1)
    print(f"\r🏥 回答: {response1}\n")

    print("-" * 30)

    # 测试问题 2 (测试紧急情况)
    query2 = "如果遇到心脏病发作的紧急情况，应该怎么处理？"
    print(f"👤 提问: {query2}")
    print("🤖 思考中...", end="", flush=True)
    response2 = rag_chain.invoke(query2)
    print(f"\r🏥 回答: {response2}\n")

    print("-" * 50)
    print("✅ 实验演示结束")


if __name__ == "__main__":
    main()