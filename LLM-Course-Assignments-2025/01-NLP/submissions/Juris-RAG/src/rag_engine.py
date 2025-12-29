"""
Juris-RAG 核心引擎模块
支持多轮对话、长上下文、引用来源显示、拒绝不确定回答
"""
import os
from typing import List, Dict, Tuple, Optional, Generator
from dataclasses import dataclass

from langchain_community.vectorstores import Chroma
try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:  # fallback for older installs
    from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
try:
    from langchain.chains import create_history_aware_retriever, create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:  # langchain>=1.0 moved legacy chains to langchain_classic
    from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
try:
    from langchain_core.documents import Document
except ImportError:  # fallback for older langchain versions
    try:
        from langchain.schema import Document
    except ImportError:
        from langchain_classic.schema import Document

# 导入配置
try:
    from src.config import (
        DB_PATH, EMBEDDING_MODEL, LLM_MODEL, SILICONFLOW_API_KEY,
        SILICONFLOW_BASE_URL, RETRIEVAL_TOP_K, RETRIEVAL_SCORE_THRESHOLD,
        LLM_TEMPERATURE, LLM_MAX_TOKENS, MAX_HISTORY_TURNS,
        CONFIDENCE_THRESHOLD, UNCERTAIN_RESPONSE
    )
except ImportError:
    # 默认回退配置
    DB_PATH = "./data/vector_db"
    EMBEDDING_MODEL = "BAAI/bge-m3"
    LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
    SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
    SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
    RETRIEVAL_TOP_K = 5
    RETRIEVAL_SCORE_THRESHOLD = 0.3
    LLM_TEMPERATURE = 0.1
    LLM_MAX_TOKENS = 2048
    MAX_HISTORY_TURNS = 10
    CONFIDENCE_THRESHOLD = 0.4
    UNCERTAIN_RESPONSE = "根据现有法律数据库，我无法回答此问题。"


@dataclass
class Citation:
    """引用来源数据类"""
    source: str
    doc_type: str
    content: str
    relevance_score: float
    metadata: Dict


@dataclass
class RAGResponse:
    """RAG响应数据类"""
    answer: str
    citations: List[Citation]
    confidence: float
    is_uncertain: bool
    retrieved_docs: List[Document]


class JurisRAGEngine:
    """法律RAG引擎"""
    
    def __init__(self, streaming: bool = True):
        """
        初始化RAG引擎
        
        Args:
            streaming: 是否启用流式输出
        """
        if not SILICONFLOW_API_KEY:
            raise ValueError("❌ 未找到 SILICONFLOW_API_KEY，请检查环境变量！")
        
        self.streaming = streaming
        self.chat_history: List[Tuple[str, str]] = []
        
        # 初始化组件
        self._init_embeddings()
        self._init_vectorstore()
        self._init_llm()
        self._init_chains()
    
    def _init_embeddings(self):
        """初始化Embedding模型"""
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_base=SILICONFLOW_BASE_URL,
            openai_api_key=SILICONFLOW_API_KEY
        )
    
    def _init_vectorstore(self):
        """初始化向量数据库"""
        if not os.path.exists(DB_PATH):
            raise FileNotFoundError(
                f"❌ 向量库不存在: {DB_PATH}\n"
                f"   请先运行: python -m src.data_processing"
            )
        
        self.vectorstore = Chroma(
            persist_directory=DB_PATH,
            embedding_function=self.embeddings
        )
        
        # 配置检索器
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": RETRIEVAL_TOP_K,
                "score_threshold": RETRIEVAL_SCORE_THRESHOLD
            }
        )
    
    def _init_llm(self):
        """初始化大语言模型"""
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            openai_api_base=SILICONFLOW_BASE_URL,
            openai_api_key=SILICONFLOW_API_KEY,
            streaming=self.streaming
        )
    
    def _init_chains(self):
        """初始化RAG链"""
        # 1. 历史对话重写链 - 将依赖上下文的问题改写为独立问题
        contextualize_q_system_prompt = """你是一个专业的问题改写助手。给定一段聊天历史和用户最新的问题，
请判断该问题是否引用了历史信息（如"它"、"这个案子"、"上面提到的"等）。

如果是，请将问题重写为一个独立的、无需上下文即可理解的问题。
如果问题已经是独立的，请原样返回。

只输出重写后的问题，不要解释。"""
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        
        self.history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )
        
        # 2. 法律问答链 - 核心Prompt
        qa_system_prompt = """你是"法律智能助手"，一个专业的中国法律问答AI。你的职责是基于检索到的法律文档，为用户提供准确、专业的法律咨询。

【核心原则】
1. **严格基于证据**：只能根据【检索到的上下文】中的信息回答，绝不编造或推测
2. **明确引用来源**：每个重要论述后必须标注来源，格式为 [来源X]
3. **承认不确定性**：如果检索内容不足以回答问题，必须明确说明

【回答格式要求】
1. 先给出直接回答（1-2句话概括）
2. 再分点详细说明（如涉及法条，逐条引用；如涉及案例，说明判例）
3. 最后给出注意事项或建议

【引用格式】
- 法条引用：根据《刑法》第X条规定，... [来源1]
- 案例引用：在类似案例中，... [来源2]

【特殊情况处理】
- 如果问题超出法律范围，礼貌说明并建议咨询专业律师
- 如果检索结果不相关或不充分，直接说"根据现有法律数据库，无法准确回答此问题"
- 不要编造不存在的法条或案例

【检索到的上下文】
{context}"""
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        
        # 3. 组合最终RAG链
        self.rag_chain = create_retrieval_chain(
            self.history_aware_retriever,
            question_answer_chain
        )
    
    def _format_chat_history(self) -> List:
        """格式化聊天历史为LangChain消息格式"""
        messages = []
        for human, ai in self.chat_history[-MAX_HISTORY_TURNS:]:
            messages.append(HumanMessage(content=human))
            messages.append(AIMessage(content=ai))
        return messages
    
    def _extract_citations(self, docs: List[Document]) -> List[Citation]:
        """从检索文档中提取引用信息"""
        citations = []
        for i, doc in enumerate(docs):
            citation = Citation(
                source=doc.metadata.get("source", "未知来源"),
                doc_type=doc.metadata.get("type", "unknown"),
                content=doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                relevance_score=doc.metadata.get("relevance_score", 0.0),
                metadata=doc.metadata
            )
            citations.append(citation)
        return citations
    
    def _calculate_confidence(self, docs: List[Document]) -> float:
        """计算回答置信度"""
        if not docs:
            return 0.0
        
        # 基于检索文档数量和相关性计算置信度
        doc_count_score = min(len(docs) / RETRIEVAL_TOP_K, 1.0)
        
        # 如果有相关性分数，使用平均分数
        scores = [doc.metadata.get("relevance_score", 0.5) for doc in docs]
        avg_score = sum(scores) / len(scores) if scores else 0.5
        
        # 综合置信度
        confidence = 0.4 * doc_count_score + 0.6 * avg_score
        return round(confidence, 2)
    
    def query(self, question: str) -> RAGResponse:
        """
        处理用户查询
        
        Args:
            question: 用户问题
            
        Returns:
            RAGResponse: 包含答案、引用、置信度等信息
        """
        # 格式化历史对话
        chat_history = self._format_chat_history()
        
        # 调用RAG链
        response = self.rag_chain.invoke({
            "input": question,
            "chat_history": chat_history
        })
        
        answer = response.get("answer", "")
        docs = response.get("context", [])
        
        # 提取引用
        citations = self._extract_citations(docs)
        
        # 计算置信度
        confidence = self._calculate_confidence(docs)
        
        # 判断是否为不确定回答
        is_uncertain = confidence < CONFIDENCE_THRESHOLD or len(docs) == 0
        
        # 如果置信度过低，使用不确定回答模板
        if is_uncertain and len(docs) == 0:
            answer = UNCERTAIN_RESPONSE
        
        # 更新对话历史
        self.chat_history.append((question, answer))
        
        return RAGResponse(
            answer=answer,
            citations=citations,
            confidence=confidence,
            is_uncertain=is_uncertain,
            retrieved_docs=docs
        )
    
    def query_stream(self, question: str) -> Generator[str, None, RAGResponse]:
        """
        流式处理用户查询
        
        Args:
            question: 用户问题
            
        Yields:
            str: 答案片段
            
        Returns:
            RAGResponse: 完整响应（在生成器结束时）
        """
        chat_history = self._format_chat_history()
        
        # 先获取检索结果
        docs = self.history_aware_retriever.invoke({
            "input": question,
            "chat_history": chat_history
        })
        
        citations = self._extract_citations(docs)
        confidence = self._calculate_confidence(docs)
        is_uncertain = confidence < CONFIDENCE_THRESHOLD or len(docs) == 0
        
        if is_uncertain and len(docs) == 0:
            yield UNCERTAIN_RESPONSE
            self.chat_history.append((question, UNCERTAIN_RESPONSE))
            return RAGResponse(
                answer=UNCERTAIN_RESPONSE,
                citations=[],
                confidence=0.0,
                is_uncertain=True,
                retrieved_docs=[]
            )
        
        # 流式生成回答
        full_answer = ""
        for chunk in self.rag_chain.stream({
            "input": question,
            "chat_history": chat_history
        }):
            if "answer" in chunk:
                full_answer += chunk["answer"]
                yield chunk["answer"]
        
        self.chat_history.append((question, full_answer))
        
        return RAGResponse(
            answer=full_answer,
            citations=citations,
            confidence=confidence,
            is_uncertain=is_uncertain,
            retrieved_docs=docs
        )
    
    def clear_history(self):
        """清空对话历史"""
        self.chat_history = []
    
    def get_history(self) -> List[Tuple[str, str]]:
        """获取对话历史"""
        return self.chat_history.copy()
    
    def search_similar(self, query: str, k: int = 5) -> List[Document]:
        """
        直接搜索相似文档（不经过LLM）
        
        Args:
            query: 搜索查询
            k: 返回数量
            
        Returns:
            List[Document]: 相似文档列表
        """
        return self.vectorstore.similarity_search(query, k=k)


# 便捷函数：获取默认RAG引擎实例
_default_engine: Optional[JurisRAGEngine] = None

def get_rag_engine(streaming: bool = True) -> JurisRAGEngine:
    """获取RAG引擎单例"""
    global _default_engine
    if _default_engine is None:
        _default_engine = JurisRAGEngine(streaming=streaming)
    return _default_engine


def get_retriever():
    """兼容旧接口：获取检索器"""
    engine = get_rag_engine()
    return engine.retriever


def get_rag_chain():
    """兼容旧接口：获取RAG链"""
    engine = get_rag_engine()
    return engine.rag_chain


# --- 命令行测试代码 ---
if __name__ == "__main__":
    print("🚀 正在初始化 Juris-RAG 引擎...")
    
    try:
        engine = JurisRAGEngine(streaming=False)
        print("✅ 引擎初始化成功！\n")
        
        # 测试问题列表
        test_questions = [
            "故意杀人罪怎么判刑？",
            "如果是情节较轻的呢？",  # 测试多轮对话
            "盗窃罪的量刑标准是什么？"
        ]
        
        for q in test_questions:
            print(f"👤 用户: {q}")
            print("-" * 50)
            
            response = engine.query(q)
            
            print(f"🤖 助手: {response.answer}")
            print(f"\n📊 置信度: {response.confidence}")
            print(f"❓ 不确定回答: {response.is_uncertain}")
            
            if response.citations:
                print("\n📚 引用来源:")
                for i, citation in enumerate(response.citations, 1):
                    print(f"   [{i}] {citation.source} ({citation.doc_type})")
                    if citation.metadata.get("accusation"):
                        print(f"       罪名: {citation.metadata['accusation']}")
            
            print("\n" + "=" * 60 + "\n")
            
    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        print("请先运行数据处理脚本构建向量库。")
    except Exception as e:
        print(f"❌ 发生错误: {e}")
