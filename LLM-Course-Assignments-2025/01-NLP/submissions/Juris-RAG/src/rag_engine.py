"""
Juris-RAG 核心引擎模块
支持多轮对话、长上下文、引用来源显示、拒绝不确定回答
"""
import os
import re
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
    LLM_MODEL = "Qwen/Qwen3-8B"
    SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
    SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
    RETRIEVAL_TOP_K = 8
    RETRIEVAL_SCORE_THRESHOLD = 0.2
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
        
        # 使用更宽松的相似度检索，后续通过后处理过滤
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": RETRIEVAL_TOP_K * 2  # 检索更多，后处理筛选
            }
        )
    
    def _extract_crime_keywords(self, query: str) -> List[str]:
        """
        从查询中提取罪名关键词，返回多个增强查询
        """
        # 常见罪名关键词映射 - 包含条款号和核心描述词
        crime_mappings = {
            "故意杀人": ["第二百三十二条 故意杀人 死刑", "侵犯公民人身权利 故意杀人"],
            "杀人": ["第二百三十二条 故意杀人 死刑", "侵犯公民人身权利"],
            "故意伤害": ["第二百三十四条 故意伤害 轻伤 重伤"],
            "伤害": ["第二百三十四条 故意伤害"],
            "盗窃": ["第二百六十四条 盗窃 数额较大 侵犯财产罪"],
            "抢劫": ["第二百六十三条 抢劫 暴力 侵犯财产罪"],
            "诈骗": ["第二百六十六条 诈骗 数额较大"],
            "正当防卫": ["第二十条 正当防卫 防卫过当 不负刑事责任"],
            "防卫": ["第二十条 正当防卫 防卫过当"],
            "自首": ["第六十七条 自首 从轻处罚 减轻处罚"],
            "累犯": ["第六十五条 累犯 从重处罚"],
            "未成年": ["第十七条 未成年人 刑事责任年龄"],
            "交通肇事": ["第一百三十三条 交通肇事 逃逸 危害公共安全"],
            "醉驾": ["第一百三十三条之一 危险驾驶"],
            "危险驾驶": ["第一百三十三条之一 危险驾驶 醉酒"],
            "贪污": ["第三百八十二条 贪污 国家工作人员"],
            "受贿": ["第三百八十五条 受贿 国家工作人员"],
            "毒品": ["第三百四十七条 走私 贩卖 运输 制造毒品"],
            "强奸": ["第二百三十六条 强奸 暴力"],
            "绑架": ["第二百三十九条 绑架 勒索财物"],
            "抢夺": ["第二百六十七条 抢夺"],
            "敲诈勒索": ["第二百七十四条 敲诈勒索"],
            "侵占": ["第二百七十条 侵占"],
        }
        
        enhanced_queries = [query]  # 原始查询始终保留
        
        for keyword, expansions in crime_mappings.items():
            if keyword in query:
                enhanced_queries.extend(expansions)
                break
        
        return enhanced_queries
    
    def _hybrid_retrieve(self, query: str, k: int = RETRIEVAL_TOP_K) -> List[Document]:
        """
        混合检索策略：分别检索法条和案例，然后合并
        使用多查询增强 + 关键词过滤提高法条检索精度
        
        ChromaDB的分数越低表示越相关（L2距离）
        """
        statute_docs = []
        case_docs = []
        seen_doc_ids = set()
        
        statute_k = max(4, k // 2 + 1)  # 法条数量
        case_k = k - statute_k + 2  # 案例数量
        
        # 1. 法条检索：使用多个增强查询
        enhanced_queries = self._extract_crime_keywords(query)
        
        for eq in enhanced_queries:
            try:
                results = self.vectorstore.similarity_search_with_score(
                    eq,
                    k=statute_k * 2,
                    filter={"type": "statute"}
                )
                
                for doc, score in results:
                    doc_id = doc.metadata.get("doc_id", id(doc))
                    if doc_id not in seen_doc_ids:
                        seen_doc_ids.add(doc_id)
                        doc.metadata["relevance_score"] = score
                        statute_docs.append(doc)
                        
            except Exception as e:
                print(f"[警告] 法条检索失败: {e}")
        
        # 2. 案例检索：使用原始查询
        try:
            case_results = self.vectorstore.similarity_search_with_score(
                query, 
                k=case_k * 2,
                filter={"type": "case"}
            )
            
            for doc, score in case_results:
                doc_id = doc.metadata.get("doc_id", id(doc))
                if doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    doc.metadata["relevance_score"] = score
                    case_docs.append(doc)
                    
        except Exception as e:
            print(f"[警告] 案例检索失败: {e}")
        
        # 3. 关键词重排序：优先包含查询关键词的法条
        def get_keyword_score(doc):
            """计算关键词匹配得分"""
            base_score = doc.metadata.get("relevance_score", 999)
            content = doc.page_content.lower()
            
            # 提取查询中的关键词
            keywords = ["故意杀人", "盗窃", "抢劫", "诈骗", "正当防卫", "自首", 
                       "交通肇事", "故意伤害", "强奸", "绑架", "未成年"]
            
            bonus = 0
            for kw in keywords:
                if kw in query and kw in content:
                    bonus -= 1.0  # 大幅提高排名（降低分数）
                    break
            
            # 额外奖励：查询中的数字词（如"232条"）
            article_nums = re.findall(r'第[一二三四五六七八九十百千零\d]+条', query)
            for num in article_nums:
                if num in content:
                    bonus -= 2.0  # 精确条款匹配，最高优先级
            
            return base_score + bonus
        
        # 对法条进行重排序
        statute_docs.sort(key=get_keyword_score)
        case_docs.sort(key=lambda d: d.metadata.get("relevance_score", 999))
        
        # 4. 合并结果：法条优先
        final_docs = []
        final_docs.extend(statute_docs[:statute_k])
        final_docs.extend(case_docs[:case_k])
        
        # 5. 回退检索
        if len(final_docs) == 0:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            for doc, score in results:
                doc.metadata["relevance_score"] = score
                final_docs.append(doc)
        
        return final_docs[:k]
    
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
        
        # 2. 法律问答链 - 核心Prompt（优化版）
        qa_system_prompt = """你是"法律智能助手"，一个专业的中国刑法问答AI。基于检索到的法条和案例回答问题。

【回答原则】
1. **优先使用法条**：如果检索到了《刑法》条文，必须优先引用法条内容
2. **案例作为补充**：案例用于说明实际判决情况，但不能替代法条
3. **如实作答**：只基于检索内容回答，无相关内容则明确说明

【回答格式】
**直接回答**：先用1-2句话概括答案（基于法条）

**法律依据**：
- 引用检索到的法条原文，标注[来源X]
- 如有多个相关条款，分别列出

**案例参考**（如有）：
- 简要说明相关案例的判决结果

**提示**：说明可能的局限或建议

【重要规则】
- 检索到法条内容时，必须直接引用原文
- 法条编号和内容必须与检索文档完全一致
- 如果检索内容不包含问题所问的罪名/情况，请直接说明"检索内容中未找到相关规定"
- 不要编造或推测法条内容

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
            # 尝试从多个字段获取相似度分数
            score = (
                doc.metadata.get("relevance_score") or
                doc.metadata.get("score") or
                doc.metadata.get("_score") or
                0.7  # 默认给予中等相关性
            )
            
            # 改进来源标注 - 包含更多元数据信息
            source_parts = [doc.metadata.get("source", "未知来源")]
            
            # 添加类型信息
            doc_type = doc.metadata.get("type", "unknown")
            if doc_type == "statute":
                article = doc.metadata.get("article", "")
                if article:
                    source_parts.append(f"({article})")
            elif doc_type == "case":
                accusation = doc.metadata.get("accusation", "")
                case_id = doc.metadata.get("case_id", "")
                if accusation:
                    source_parts.append(f"【{accusation}】")
                if case_id:
                    source_parts.append(f"(案号:{case_id})")
            
            source_display = "".join(source_parts)
            
            citation = Citation(
                source=source_display,
                doc_type=doc_type,
                content=doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                relevance_score=float(score),
                metadata=doc.metadata
            )
            citations.append(citation)
        return citations

    def _attach_similarity_scores(self, query: str, docs: List[Document]) -> List[Document]:
        """为检索到的文档补充相似度分数。"""
        if not docs:
            return docs
        try:
            # 预取更多候选，以便覆盖 context 中的文档
            k = max(len(docs), RETRIEVAL_TOP_K * 2)
            scored = self.vectorstore.similarity_search_with_score(query, k=k)
            score_map = {}
            for d, score in scored:
                doc_id = d.metadata.get("doc_id")
                if doc_id:
                    score_map[doc_id] = score
            for doc in docs:
                doc_id = doc.metadata.get("doc_id")
                if doc_id and doc_id in score_map:
                    doc.metadata["relevance_score"] = score_map[doc_id]
            return docs
        except Exception:
            return docs
    
    def _calculate_confidence(self, docs: List[Document]) -> float:
        """计算回答置信度"""
        if not docs:
            return 0.0
        
        # ChromaDB的分数越低越相关，需要转换
        scores = []
        has_statute = False
        
        for doc in docs:
            raw_score = doc.metadata.get("relevance_score", 1.0)
            # 转换为0-1的相关性分数（分数越低越相关）
            relevance = max(0, 1 - raw_score / 2)
            scores.append(relevance)
            
            if doc.metadata.get("type") == "statute":
                has_statute = True
        
        max_score = max(scores) if scores else 0.0
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # 如果有法条文档，置信度提升
        statute_bonus = 0.1 if has_statute else 0
        
        # 综合计算置信度
        confidence = 0.4 * max_score + 0.4 * avg_score + 0.2 * min(len(docs) / RETRIEVAL_TOP_K, 1.0) + statute_bonus
        
        return round(min(confidence, 0.95), 2)
    
    def query(self, question: str) -> RAGResponse:
        """
        处理用户查询（使用混合检索策略）
        
        Args:
            question: 用户问题
            
        Returns:
            RAGResponse: 包含答案、引用、置信度等信息
        """
        # 格式化历史对话
        chat_history = self._format_chat_history()
        
        # 使用混合检索策略获取文档
        docs = self._hybrid_retrieve(question, k=RETRIEVAL_TOP_K)
        
        # 如果没有检索到有效文档
        if not docs:
            self.chat_history.append((question, UNCERTAIN_RESPONSE))
            return RAGResponse(
                answer=UNCERTAIN_RESPONSE,
                citations=[],
                confidence=0.0,
                is_uncertain=True,
                retrieved_docs=[]
            )
        
        # 构建上下文
        context_parts = []
        for i, doc in enumerate(docs, 1):
            doc_type = doc.metadata.get("type", "unknown")
            source = doc.metadata.get("source", "未知来源")
            
            if doc_type == "statute":
                article = doc.metadata.get("article", "")
                context_parts.append(f"[来源{i}] 【法条】{source} {article}\n{doc.page_content}")
            else:
                accusation = doc.metadata.get("accusation", "")
                context_parts.append(f"[来源{i}] 【案例】{source}（{accusation}）\n{doc.page_content}")
        
        context_text = "\n\n".join(context_parts)
        
        # 直接调用LLM生成回答
        from langchain_core.messages import HumanMessage, SystemMessage
        
        qa_prompt = f"""你是"法律智能助手"，一个专业的中国刑法问答AI。基于检索到的法条和案例回答问题。

【回答原则】
1. **优先使用法条**：如果检索到了《刑法》条文，必须优先引用法条内容
2. **案例作为补充**：案例用于说明实际判决情况，但不能替代法条
3. **如实作答**：只基于检索内容回答，无相关内容则明确说明

【回答格式】
**直接回答**：先用1-2句话概括答案（基于法条）

**法律依据**：
- 引用检索到的法条原文，标注[来源X]
- 如有多个相关条款，分别列出

**案例参考**（如有相关案例）：
- 简要说明相关案例的判决结果

**提示**：说明可能的局限或建议

【重要规则】
- 检索到法条内容时，必须直接引用原文
- 法条编号和内容必须与检索文档完全一致
- 如果检索内容不包含问题所问的罪名/情况，请直接说明"检索内容中未找到相关规定"

【检索到的上下文】
{context_text}

【用户问题】
{question}"""
        
        messages = [HumanMessage(content=qa_prompt)]
        
        response = self.llm.invoke(messages)
        answer = response.content
        
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
        docs = self._attach_similarity_scores(question, docs)
        
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
