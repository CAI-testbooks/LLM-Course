import os
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import FAISS
from .rag import init_embeddings, load_and_process_pubmed, split_documents, build_vector_db
# 导入独立的百度翻译模块
from .translate import BaiduGeneralTranslator


class DeepSeekAPI:
    def __init__(self):
        # 初始化DeepSeek客户端
        self.client = OpenAI(
            api_key=os.environ.get('DEEPSEEK_API_KEY', 'sk-96317155510849dfbe5faa9e6ba2bf35'),
            base_url="https://api.deepseek.com"
        )

        # 系统提示
        self.system_prompt = """你是专业的医学智能问答助手，需严格按以下规则回答：
            1. 仅回答医学、健康相关问题，其他问题直接拒绝，回复“抱歉，我仅能解答医学领域相关问题，请提问医学相关内容”。
            2. 若提供的“参考文献”为“（无相关文献）”，直接基于医学常识回答，**不输出任何参考文献**。
            3. 若提供了相关文献，优先基于文献内容回答，语言通俗易懂，**不输出任何参考文献**。 
            4. 所有回答必须使用中文。
            """

        # 核心参数
        self.similarity_threshold = 0.4  # 相似度阈值

        # 路径配置
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_path = os.path.join(self.current_dir, "../pubmed_faiss_db")
        self.json_path = os.path.join(self.current_dir, "../pubmed.json")

        # 初始化百度医学翻译器
        self.translator = self.init_translator()

        # 初始化向量库和嵌入模型
        self.init_vector_db()

    # 初始化翻译器
    def init_translator(self):
        try:
            return BaiduGeneralTranslator()
        except Exception as e:
            print(f"百度翻译器初始化失败：{e}，将使用原始文本检索")
            return None

    # 向量数据库初始化
    def init_vector_db(self):
        if not os.path.exists(f"{self.db_path}/index.faiss") or not os.path.exists(f"{self.db_path}/index.pkl"):
            print(f"未找到向量数据库，开始构建（路径：{self.db_path}）...")
            self.embeddings = init_embeddings(device="cuda" if os.environ.get("USE_CUDA") else "cpu")
            if not os.path.exists(self.json_path):
                raise FileNotFoundError(f"未找到数据文件: {self.json_path}，请确保该文件存在")
            documents = load_and_process_pubmed(self.json_path)
            chunks = split_documents(documents)
            self.db = build_vector_db(chunks, self.embeddings, self.db_path)
        else:
            self.embeddings = init_embeddings(device="cuda" if os.environ.get("USE_CUDA") else "cpu")
            self.db = FAISS.load_local(self.db_path, self.embeddings, allow_dangerous_deserialization=True)
        print(f"向量数据库初始化完成")

    # 调用翻译器
    def translate_medical_query(self, query):
        if not self.translator:
            return query  # 翻译器未初始化，返回原始文本
        return self.translator.translate(query)

    # 余弦相似度检索并去重
    def retrieve_relevant_docs(self, query, k=5):
        translated_query = self.translate_medical_query(query)
        candidate_docs = self.db.similarity_search(translated_query, k=10)
        if not candidate_docs:
            return []

        query_embedding = self.embeddings.embed_query(translated_query)
        relevant_docs = []
        for doc in candidate_docs:
            doc_embedding = self.embeddings.embed_query(doc.page_content.strip())
            similarity = cosine_similarity(
                np.array([query_embedding]),
                np.array([doc_embedding])
            )[0][0]

            print(f"文献相似度：{round(similarity, 3)}")
            if similarity >= self.similarity_threshold:
                relevant_docs.append((doc, similarity))

        # 按文献去重
        doc_unique = {}
        for doc, sim in relevant_docs:
            doc_key = f"{doc.metadata['title']}_{doc.metadata['source']}"
            if doc_key not in doc_unique or sim > doc_unique[doc_key][1]:
                doc_unique[doc_key] = (doc, sim)

        # 排序后返回
        sorted_docs = sorted(doc_unique.values(), key=lambda x: x[1], reverse=True)
        return [doc for doc, sim in sorted_docs[:k]]

    # 生成响应
    def get_response(self, chat_history):
        try:
            user_query = chat_history[-1]["content"].strip()
            if not user_query:
                return "请输入有效的医学问题"

            relevant_docs = self.retrieve_relevant_docs(user_query, k=3)

            context = ""
            sources = ""
            if relevant_docs:
                context = "\n\n".join([
                    f"文献标题：{doc.metadata['title']}\n内容：{doc.page_content[:600]}"
                    for doc in relevant_docs
                ])
                sources = "\n".join([f"- {doc.metadata['title']}" for doc in relevant_docs])
            else:
                context = "（无相关文献）"

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"用户问题：{user_query}\n参考文献：{context}"}
            ]
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=False
            )
            answer = response.choices[0].message.content

            if relevant_docs:
                return f"{answer}\n\n参考文献：\n{sources}"
            else:
                return f"{answer}\n\n未检索到相关医学文献"

        except Exception as e:
            return f"调用API出错：{str(e)}"
