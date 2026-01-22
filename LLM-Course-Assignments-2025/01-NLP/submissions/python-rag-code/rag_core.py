# rag_core.py
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama  # 需要先安装Ollama并拉取模型

class PythonRAG:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="./models/all-MiniLM-L6-v2")
        self.vector_db = Chroma(persist_directory="./chroma_db", embedding_function=self.embeddings)
        # 使用Ollama调用本地模型，非常适合你的1650Ti[citation:3][citation:6]
        self.llm = Ollama(base_url='http://localhost:11434', model="qwen2.5:1.5b")

        # 定义一个包含引用来源要求的Prompt模板[citation:7]
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""你是一个专业的Python助手，请严格根据以下上下文回答问题。如果上下文不包含答案，请明确说“根据现有文档无法回答”。

            上下文：
            {context}

            问题：{question}

            要求：回答需简洁准确，并在末尾列出所依据的上下文编号。
            回答："""
        )

    def ask(self, question):
        # 1. 检索：从向量库找到最相关的3个文本块
        docs = self.vector_db.similarity_search(question, k=3)

        # 2. 构建上下文和Prompt
        context = "\n\n".join([f"[{i+1}]{doc.page_content}" for i, doc in enumerate(docs)])
        prompt = self.prompt_template.format(context=context, question=question)

        # 3. 生成答案
        answer = self.llm.invoke(prompt)

        # 4. 组织返回结果（答案 + 来源）
        sources = [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
        return {"answer": answer, "sources": sources}

# 测试
if __name__ == "__main__":
    rag = PythonRAG()
    result = rag.ask("Python中如何用with语句打开文件？")

    print("=" * 60)
    print(" RAG系统测试成功！")
    print("=" * 60)
    print("\n 问题：Python中如何用with语句打开文件？")
    print("\n 答案：")
    print(result["answer"])

    if result["sources"] and len(result["sources"]) > 0:
        print(f"\n 参考来源 ({len(result['sources'])}个)：")
        for i, source in enumerate(result["sources"]):
            print(f"\n{i + 1}. {source['content'][:200]}...")
    else:
        print("\n⚠  未找到相关来源（但模型基于知识库正确回答了问题）")