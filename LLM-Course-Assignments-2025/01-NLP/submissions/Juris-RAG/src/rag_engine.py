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

# 尝试导入链组件（兼容不同版本）
try:
    from langchain.chains import create_history_aware_retriever, create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    try:
        from langchain_core.runnables import RunnablePassthrough
        # 如果没有传统chains，使用简化实现
        create_history_aware_retriever = None
        create_retrieval_chain = None
        create_stuff_documents_chain = None
    except ImportError:
        pass

try:
    from langchain_core.documents import Document
except ImportError:  # fallback for older langchain versions
    try:
        from langchain.schema import Document
    except ImportError:
        from langchain_community.docstore.document import Document

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

# ==================== 方案A: 超范围检测配置 ====================
# 非刑法领域关键词 - 检测到这些词时触发超范围拒绝
OUT_OF_SCOPE_KEYWORDS = [
    # 民法相关
    "民法典", "合同法", "婚姻法", "继承法", "物权法", "侵权责任",
    "民事纠纷", "离婚", "抚养权", "遗产继承", "房产纠纷", "债务纠纷",
    "借款合同", "租赁合同", "买卖合同", "劳务合同",
    # 商法相关  
    "公司法", "证券法", "保险法", "票据法", "破产法",
    "股票", "基金", "投资理财", "上市公司", "董事会", "股东",
    "商业秘密", "知识产权", "专利", "商标", "著作权",
    # 行政法相关
    "行政处罚", "行政复议", "行政诉讼", "拆迁", "土地征收",
    "行政许可", "行政强制", "公务员", "事业编",
    # 劳动法相关
    "劳动法", "劳动合同", "社保", "工伤", "劳动仲裁",
    "加班费", "年假", "辞退赔偿", "五险一金",
    # 其他非刑法
    "税法", "海关", "环保法", "食品安全",
    "医疗纠纷", "医患关系", "交通事故赔偿"
]

# 超范围拒绝响应模板
OUT_OF_SCOPE_RESPONSE = """抱歉，您的问题涉及**{detected_domain}**领域，不在本系统的服务范围内。

**本系统专注于中国刑法领域**，包括：
- 各类刑事犯罪的认定与量刑
- 刑事责任年龄、自首、立功等情节
- 正当防卫、紧急避险等免责事由
- 刑事案例的判决参考

**建议**：
- 民事问题请咨询民事律师或查阅民法典
- 商事问题请咨询公司法/证券法律师
- 劳动问题请咨询劳动仲裁部门或劳动律师
- 行政问题请咨询行政法律师或相关政府部门

如果您有**刑法相关问题**，欢迎继续咨询！"""

# 最低相关性阈值 - 低于此值视为超范围
MIN_RELEVANCE_THRESHOLD = 0.65


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
    """法律RAG引擎 - 支持多领域"""
    
    def _try_load_multi_domain_vectorstores(self) -> bool:
        """
        尝试加载多领域向量库
        
        Returns:
            True 如果成功加载多领域，False 否则
        """
        legal_domains = {
            'criminal': '刑法',
            'civil': '民法',
            'commercial': '商法',
            'administrative': '行政法',
            'labor': '劳动法'
        }
        
        self.vectorstores_multi = {}
        loaded_count = 0
        
        for domain_key, domain_name in legal_domains.items():
            domain_db_path = os.path.join(DB_PATH, domain_key)
            if os.path.exists(domain_db_path):
                try:
                    vs = Chroma(
                        persist_directory=domain_db_path,
                        embedding_function=self.embeddings
                    )
                    self.vectorstores_multi[domain_key] = {
                        'vectorstore': vs,
                        'retriever': vs.as_retriever(search_type="similarity", search_kwargs={"k": RETRIEVAL_TOP_K * 2}),
                        'name': domain_name
                    }
                    loaded_count += 1
                except Exception as e:
                    print(f"⚠️  无法加载 {domain_name} 向量库: {e}")
        
        if loaded_count > 0:
            print(f"✅ 成功加载 {loaded_count} 个领域的向量库")
            return True
        return False
    
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
    
    # ==================== 方案A: 超范围检测 ====================
    def _detect_out_of_scope(self, query: str) -> Tuple[bool, str]:
        """
        检测问题是否超出刑法范围
        
        Args:
            query: 用户问题
            
        Returns:
            Tuple[bool, str]: (是否超范围, 检测到的领域)
        """
        query_lower = query.lower()
        
        # 检测到的非刑法领域
        detected_domains = []
        
        # 按领域分组检测
        domain_keywords = {
            "民法/合同法": ["民法典", "合同法", "婚姻法", "继承法", "物权法", "侵权责任",
                         "民事纠纷", "离婚", "抚养权", "遗产继承", "房产纠纷", "债务纠纷",
                         "借款合同", "租赁合同", "买卖合同", "劳务合同"],
            "商法/公司法": ["公司法", "证券法", "保险法", "票据法", "破产法",
                         "股票", "基金", "投资理财", "上市公司", "董事会", "股东",
                         "商业秘密", "知识产权", "专利", "商标", "著作权"],
            "行政法": ["行政处罚", "行政复议", "行政诉讼", "拆迁", "土地征收",
                     "行政许可", "行政强制", "公务员", "事业编"],
            "劳动法": ["劳动法", "劳动合同", "社保", "工伤", "劳动仲裁",
                     "加班费", "年假", "辞退赔偿", "五险一金"],
            "其他非刑法": ["税法", "海关", "环保法", "食品安全",
                        "医疗纠纷", "医患关系", "交通事故赔偿"]
        }
        
        for domain, keywords in domain_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    detected_domains.append(domain)
                    break
        
        if detected_domains:
            # 去重并返回第一个检测到的领域
            return True, detected_domains[0]
        
        return False, ""
    
    def _is_low_relevance(self, docs: List[Document]) -> bool:
        """
        检测检索结果的相关性是否过低
        如果所有文档的相关性都低于阈值，认为超出范围
        
        Args:
            docs: 检索到的文档
            
        Returns:
            bool: 是否相关性过低
        """
        if not docs:
            return True
        
        # 计算法条文档的平均相关性
        statute_scores = []
        for doc in docs:
            if doc.metadata.get("type") == "statute":
                score = doc.metadata.get("relevance_score", 1.0)
                # ChromaDB分数越低越相关
                relevance = 1 - min(score / 2, 1.0)
                statute_scores.append(relevance)
        
        if not statute_scores:
            # 没有检索到法条，可能是超范围
            return True
        
        avg_relevance = sum(statute_scores) / len(statute_scores)
        return avg_relevance < MIN_RELEVANCE_THRESHOLD
    
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
        
        # 尝试加载多领域向量库
        self.multi_domain_mode = self._try_load_multi_domain_vectorstores()
        
        if self.multi_domain_mode:
            print("✅ 多领域模式已启动")
        else:
            # 回退到单领域模式（刑法）
            print("⚠️  未检测到多领域向量库，使用单领域模式（刑法）")
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
        方案C增强：扩展罪名映射，支持更多罪名
        """
        # 常见罪名关键词映射 - 包含条款号和核心描述词（扩展版）
        crime_mappings = {
            # 侵犯公民人身权利罪
            "故意杀人": ["第二百三十二条 故意杀人 死刑 无期徒刑", "侵犯公民人身权利 故意杀人"],
            "杀人": ["第二百三十二条 故意杀人 死刑", "侵犯公民人身权利"],
            "故意伤害": ["第二百三十四条 故意伤害 轻伤 重伤 致人死亡"],
            "伤害": ["第二百三十四条 故意伤害"],
            "强奸": ["第二百三十六条 强奸 暴力 胁迫 妇女", "侵犯公民人身权利 强奸罪"],
            "绑架": ["第二百三十九条 绑架 勒索财物 人质"],
            "拐卖": ["第二百四十条 拐卖妇女儿童"],
            "非法拘禁": ["第二百三十八条 非法拘禁 剥夺人身自由"],
            
            # 侵犯财产罪
            "盗窃": ["第二百六十四条 盗窃 数额较大 数额巨大 侵犯财产罪"],
            "抢劫": ["第二百六十三条 抢劫 暴力 胁迫 侵犯财产罪"],
            "诈骗": ["第二百六十六条 诈骗 数额较大 数额巨大"],
            "抢夺": ["第二百六十七条 抢夺 公然夺取"],
            "敲诈勒索": ["第二百七十四条 敲诈勒索 威胁 要挟"],
            "侵占": ["第二百七十条 侵占 代为保管"],
            "挪用": ["第二百七十二条 挪用资金"],
            
            # 危害公共安全罪
            "交通肇事": ["第一百三十三条 交通肇事 逃逸 重大事故 危害公共安全"],
            "醉驾": ["第一百三十三条之一 危险驾驶 醉酒驾驶"],
            "危险驾驶": ["第一百三十三条之一 危险驾驶 醉酒 追逐竞驶"],
            "放火": ["第一百一十四条 放火罪 危害公共安全"],
            "爆炸": ["第一百一十四条 爆炸罪 危害公共安全"],
            
            # 妨害社会管理秩序罪
            "聚众斗殴": ["第二百九十二条 聚众斗殴 首要分子 积极参加"],
            "寻衅滋事": ["第二百九十三条 寻衅滋事 随意殴打 追逐拦截"],
            "赌博": ["第三百零三条 赌博罪 开设赌场"],
            "伪证": ["第三百零五条 伪证罪 虚假证明 证人"],
            "包庇": ["第三百一十条 包庇罪 窝藏 隐瞒"],
            "妨害公务": ["第二百七十七条 妨害公务 暴力 威胁"],
            
            # 贪污贿赂罪
            "贪污": ["第三百八十二条 贪污罪 国家工作人员 侵吞"],
            "受贿": ["第三百八十五条 受贿罪 国家工作人员 谋取利益"],
            "行贿": ["第三百八十九条 行贿罪 给予财物"],
            "挪用公款": ["第三百八十四条 挪用公款 归个人使用"],
            
            # 走私贩毒罪
            "毒品": ["第三百四十七条 走私贩卖运输制造毒品 走私罪"],
            "贩毒": ["第三百四十七条 贩卖毒品 走私运输制造"],
            "走私": ["第一百五十一条 走私罪 武器弹药 核材料 假币", "第一百五十三条 走私普通货物"],
            
            # 刑罚制度
            "正当防卫": ["第二十条 正当防卫 防卫过当 不负刑事责任 不法侵害"],
            "防卫": ["第二十条 正当防卫 防卫过当"],
            "紧急避险": ["第二十一条 紧急避险 避免危险"],
            "自首": ["第六十七条 自首 从轻处罚 减轻处罚 自动投案"],
            "立功": ["第六十八条 立功 重大立功 减轻处罚"],
            "累犯": ["第六十五条 累犯 从重处罚 五年以内"],
            "缓刑": ["第七十二条 缓刑 宣告缓刑 三年以下有期徒刑", "第七十三条 缓刑考验期"],
            "减刑": ["第七十八条 减刑 悔改表现 立功表现"],
            "假释": ["第八十一条 假释 服刑期间 不致再危害社会"],
            "未成年": ["第十七条 未成年人 刑事责任年龄 从轻减轻"],
            "刑事责任年龄": ["第十七条 刑事责任年龄 十四周岁 十六周岁"],
            "从轻": ["第六十七条 从轻处罚", "第十七条 从轻减轻"],
            "减轻": ["第六十三条 减轻处罚 法定刑以下"],
            "从重": ["第六十五条 从重处罚"],
            "共同犯罪": ["第二十五条 共同犯罪 二人以上共同故意"],
            "主犯": ["第二十六条 主犯 组织领导 主要作用"],
            "从犯": ["第二十七条 从犯 次要辅助作用"],
        }
        
        enhanced_queries = [query]  # 原始查询始终保留
        matched_keywords = []
        
        # 匹配所有相关关键词
        for keyword, expansions in crime_mappings.items():
            if keyword in query:
                enhanced_queries.extend(expansions)
                matched_keywords.append(keyword)
        
        # 如果没有匹配到任何关键词，尝试通用法律查询增强
        if not matched_keywords:
            enhanced_queries.append(f"刑法 {query} 处罚")
            enhanced_queries.append(f"{query} 有期徒刑 罚金")
        
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
        
        # 3. 方案C增强：关键词重排序 + 语义相关性融合
        def get_keyword_score(doc):
            """计算关键词匹配得分 - 增强版"""
            base_score = doc.metadata.get("relevance_score", 999)
            content = doc.page_content.lower()
            query_lower = query.lower()
            
            # 扩展的关键词列表（覆盖更多罪名）
            crime_keywords = [
                # 侵犯人身权利
                "故意杀人", "故意伤害", "强奸", "绑架", "拐卖", "非法拘禁",
                # 侵犯财产
                "盗窃", "抢劫", "诈骗", "抢夺", "敲诈勒索", "侵占", "挪用",
                # 危害公共安全
                "交通肇事", "危险驾驶", "醉驾", "放火", "爆炸",
                # 妨害社会管理
                "聚众斗殴", "寻衅滋事", "赌博", "伪证", "包庇", "妨害公务",
                # 贪污贿赂
                "贪污", "受贿", "行贿", "挪用公款",
                # 毒品犯罪
                "毒品", "贩毒", "走私",
                # 刑罚制度
                "正当防卫", "紧急避险", "自首", "立功", "累犯",
                "缓刑", "减刑", "假释", "未成年", "共同犯罪",
                "从轻", "减轻", "从重", "主犯", "从犯"
            ]
            
            bonus = 0
            matched_count = 0
            
            # 统计匹配的关键词数量
            for kw in crime_keywords:
                if kw in query_lower and kw in content:
                    matched_count += 1
                    bonus -= 0.5  # 每个匹配关键词降低0.5分
            
            # 额外奖励：多个关键词匹配
            if matched_count >= 2:
                bonus -= 0.5  # 额外奖励
            
            # 精确条款匹配（最高优先级）
            article_nums = re.findall(r'第[一二三四五六七八九十百千零\d]+条', query)
            for num in article_nums:
                if num in content:
                    bonus -= 2.0  # 精确条款匹配
            
            # 查询词直接出现在内容中
            query_words = [w for w in query_lower.split() if len(w) >= 2]
            for word in query_words:
                if word in content:
                    bonus -= 0.3
            
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
        """初始化RAG链 - 简化版（不依赖传统chains）"""
        # 由于使用了混合检索策略，不再需要传统的chains
        # 直接在query方法中处理检索和生成
        self.history_aware_retriever = None
        self.rag_chain = None
    
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
        处理用户查询（使用混合检索策略 + 超范围检测）
        
        方案A: 超范围检测 - 检测非刑法领域问题并拒绝
        方案C: 混合检索 + 重排序
        方案D: 优化提示词
        
        Args:
            question: 用户问题
            
        Returns:
            RAGResponse: 包含答案、引用、置信度等信息
        """
        # ==================== 方案A: 超范围检测 ====================
        is_out_of_scope, detected_domain = self._detect_out_of_scope(question)
        
        if is_out_of_scope:
            # 生成超范围拒绝响应
            out_of_scope_answer = OUT_OF_SCOPE_RESPONSE.format(detected_domain=detected_domain)
            self.chat_history.append((question, out_of_scope_answer))
            return RAGResponse(
                answer=out_of_scope_answer,
                citations=[],
                confidence=0.1,  # 低置信度表示这是拒绝回答
                is_uncertain=True,
                retrieved_docs=[]
            )
        
        # 格式化历史对话
        chat_history = self._format_chat_history()
        
        # 使用混合检索策略获取文档
        docs = self._hybrid_retrieve(question, k=RETRIEVAL_TOP_K)
        
        # 检测检索结果相关性是否过低（第二道防线）
        if self._is_low_relevance(docs):
            low_relevance_answer = """抱歉，在现有的刑法数据库中未找到与您问题高度相关的法条。

**可能的原因**：
1. 问题涉及的具体法律规定不在当前知识库覆盖范围内
2. 问题表述可能需要更具体的法律术语
3. 该问题可能涉及其他法律领域

**建议**：
- 请尝试使用更具体的法律术语描述问题
- 如果涉及具体案件，建议咨询专业刑事律师
- 如有其他刑法相关问题，欢迎继续咨询"""
            self.chat_history.append((question, low_relevance_answer))
            return RAGResponse(
                answer=low_relevance_answer,
                citations=[],
                confidence=0.2,
                is_uncertain=True,
                retrieved_docs=docs
            )
        
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
        
        # ==================== 方案D: 优化提示词 ====================
        qa_prompt = f"""你是"法律智能助手"，一个专业的**中国刑法**问答AI。你只回答刑法相关问题。

【系统说明】
本系统专注于中国刑法领域，包括：
- 各类刑事犯罪的认定与量刑（如故意杀人、盗窃、诈骗等）
- 刑事责任年龄、自首、立功、累犯等量刑情节
- 正当防卫、紧急避险等免责事由
- 刑事案例的判决参考

【回答原则】
1. **严格基于检索内容**：只使用检索到的法条和案例回答，不编造
2. **法条优先**：如检索到《刑法》条文，必须优先引用法条原文
3. **准确引用**：法条编号和内容必须与检索文档完全一致
4. **诚实回答**：如检索内容不包含相关规定，明确说明"未检索到相关法条"

【回答格式】
**直接回答**：用1-2句话概括核心结论（基于法条）

**法律依据**：
- 引用检索到的法条原文，标注[来源X]
- 必须包含完整的条款号（如"第二百三十二条"）

**案例参考**（如有）：
- 简要说明相关案例的判决结果

**提示**：说明注意事项或建议咨询专业律师

【检索到的上下文】
{context_text}

【用户问题】
{question}

请基于上述检索内容回答问题。如果检索内容不包含答案所需信息，请诚实说明。"""
        
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
        流式处理用户查询（使用混合检索 + 超范围检测）
        
        Args:
            question: 用户问题
            
        Yields:
            str: 答案片段
            
        Returns:
            RAGResponse: 完整响应（在生成器结束时）
        """
        # 超范围检测
        is_out_of_scope, detected_domain = self._detect_out_of_scope(question)
        
        if is_out_of_scope:
            out_of_scope_answer = OUT_OF_SCOPE_RESPONSE.format(detected_domain=detected_domain)
            yield out_of_scope_answer
            self.chat_history.append((question, out_of_scope_answer))
            return RAGResponse(
                answer=out_of_scope_answer,
                citations=[],
                confidence=0.1,
                is_uncertain=True,
                retrieved_docs=[]
            )
        
        # 使用混合检索
        docs = self._hybrid_retrieve(question, k=RETRIEVAL_TOP_K)
        
        # 检测相关性
        if self._is_low_relevance(docs):
            low_relevance_answer = """抱歉，在现有的刑法数据库中未找到与您问题高度相关的法条。建议咨询专业刑事律师。"""
            yield low_relevance_answer
            self.chat_history.append((question, low_relevance_answer))
            return RAGResponse(
                answer=low_relevance_answer,
                citations=[],
                confidence=0.2,
                is_uncertain=True,
                retrieved_docs=docs
            )
        
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
        
        # 构建提示词
        qa_prompt = f"""你是"法律智能助手"，一个专业的中国刑法问答AI。

【检索到的上下文】
{context_text}

【用户问题】
{question}

请基于检索内容回答问题，优先引用法条原文。"""
        
        # 流式生成回答
        full_answer = ""
        for chunk in self.llm.stream([HumanMessage(content=qa_prompt)]):
            if chunk.content:
                full_answer += chunk.content
                yield chunk.content
        
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
