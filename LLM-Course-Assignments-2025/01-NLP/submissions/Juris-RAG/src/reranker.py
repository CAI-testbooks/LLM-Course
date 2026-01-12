"""
Juris-RAG Reranker 模块
基于LLM的文档重排序和超范围判别器
用于：
1. 对检索结果进行相关性重排序
2. 检测超范围问题（二次验证）
3. 检测潜在幻觉风险
"""
import os
import re
import json
import hashlib
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from functools import lru_cache
from collections import OrderedDict
import time
import threading

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.documents import Document
except ImportError:
    from langchain_community.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage, Document

# 导入配置
try:
    from src.config import (
        SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL,
        RERANKER_MODEL, RERANKER_TOP_K, RERANKER_THRESHOLD,
        ENABLE_RERANKER, ENABLE_HALLUCINATION_CHECK,
        ENABLE_CACHE, CACHE_MAX_SIZE, CACHE_TTL_SECONDS,
        LLM_MODEL
    )
except ImportError:
    SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
    SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
    RERANKER_MODEL = "Qwen/Qwen3-8B"
    LLM_MODEL = "Qwen/Qwen3-8B"
    RERANKER_TOP_K = 5
    RERANKER_THRESHOLD = 0.4
    ENABLE_RERANKER = True
    ENABLE_HALLUCINATION_CHECK = True
    ENABLE_CACHE = True
    CACHE_MAX_SIZE = 100
    CACHE_TTL_SECONDS = 3600


@dataclass
class RerankedDocument:
    """重排序后的文档"""
    document: Document
    relevance_score: float
    is_relevant: bool
    reasoning: str = ""


@dataclass
class ScopeCheckResult:
    """超范围检测结果"""
    is_in_scope: bool
    confidence: float
    detected_domain: str
    reasoning: str


@dataclass
class HallucinationCheckResult:
    """幻觉检测结果"""
    has_hallucination_risk: bool
    risk_level: str  # low, medium, high
    problematic_claims: List[str]
    reasoning: str


class TTLCache:
    """带过期时间的LRU缓存"""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[str, float] = {}
        self.lock = threading.Lock()
    
    def _generate_key(self, *args) -> str:
        """生成缓存键"""
        content = json.dumps(args, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[any]:
        """获取缓存值"""
        with self.lock:
            if key not in self.cache:
                return None
            
            # 检查是否过期
            if time.time() - self.timestamps[key] > self.ttl_seconds:
                del self.cache[key]
                del self.timestamps[key]
                return None
            
            # 移到末尾（最近使用）
            self.cache.move_to_end(key)
            return self.cache[key]
    
    def set(self, key: str, value: any):
        """设置缓存值"""
        with self.lock:
            # 如果已存在，更新
            if key in self.cache:
                self.cache[key] = value
                self.timestamps[key] = time.time()
                self.cache.move_to_end(key)
                return
            
            # 如果超出大小限制，删除最旧的
            while len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]
            
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()


class LLMReranker:
    """基于LLM的重排序器和判别器"""
    
    def __init__(self):
        if not SILICONFLOW_API_KEY:
            raise ValueError("❌ 未找到 SILICONFLOW_API_KEY")
        
        self.llm = ChatOpenAI(
            model=RERANKER_MODEL if RERANKER_MODEL else LLM_MODEL,
            temperature=0.0,  # 确定性输出
            max_tokens=512,
            openai_api_base=SILICONFLOW_BASE_URL,
            openai_api_key=SILICONFLOW_API_KEY
        )
        
        # 初始化缓存
        if ENABLE_CACHE:
            self.cache = TTLCache(max_size=CACHE_MAX_SIZE, ttl_seconds=CACHE_TTL_SECONDS)
        else:
            self.cache = None
    
    def rerank_documents(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: int = None
    ) -> List[RerankedDocument]:
        """
        对检索到的文档进行重排序
        
        Args:
            query: 用户查询
            documents: 检索到的文档列表
            top_k: 保留的文档数量
            
        Returns:
            重排序后的文档列表
        """
        if not ENABLE_RERANKER or not documents:
            # 不启用重排序，直接返回原始文档
            return [
                RerankedDocument(
                    document=doc,
                    relevance_score=1.0 - min(doc.metadata.get("relevance_score", 0.5) / 2, 1.0),
                    is_relevant=True
                )
                for doc in documents
            ]
        
        top_k = top_k or RERANKER_TOP_K
        
        # 检查缓存
        if self.cache:
            cache_key = self.cache._generate_key(query, [d.page_content[:100] for d in documents[:10]])
            cached_result = self.cache.get(cache_key)
            if cached_result:
                return cached_result[:top_k]
        
        # 构建重排序提示词
        docs_text = ""
        for i, doc in enumerate(documents[:10]):  # 最多处理10个文档
            doc_type = doc.metadata.get("type", "unknown")
            source = doc.metadata.get("source", "未知")
            content = doc.page_content[:300]
            docs_text += f"[文档{i+1}] 类型:{doc_type} 来源:{source}\n{content}\n\n"
        
        prompt = f"""你是法律文档相关性评估专家。请评估以下文档与用户问题的相关性。

【用户问题】
{query}

【候选文档】
{docs_text}

【评估要求】
1. 为每个文档评分（0-10分，10分最相关）
2. 判断文档是否与问题相关（是/否）
3. 优先考虑直接回答问题的法条文档

【输出格式】（严格按JSON格式）
{{
  "rankings": [
    {{"doc_id": 1, "score": 8, "relevant": true}},
    {{"doc_id": 2, "score": 5, "relevant": true}},
    ...
  ]
}}

请只输出JSON，不要其他内容。"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            result_text = response.content.strip()
            
            # 解析JSON
            json_match = re.search(r'\{[\s\S]*\}', result_text)
            if json_match:
                result = json.loads(json_match.group())
                rankings = result.get("rankings", [])
                
                # 构建重排序结果
                reranked = []
                doc_scores = {r["doc_id"]: r for r in rankings}
                
                for i, doc in enumerate(documents[:10]):
                    score_info = doc_scores.get(i + 1, {"score": 5, "relevant": True})
                    normalized_score = score_info["score"] / 10.0
                    
                    reranked.append(RerankedDocument(
                        document=doc,
                        relevance_score=normalized_score,
                        is_relevant=score_info.get("relevant", True) and normalized_score >= RERANKER_THRESHOLD
                    ))
                
                # 按分数排序
                reranked.sort(key=lambda x: x.relevance_score, reverse=True)
                
                # 缓存结果
                if self.cache:
                    self.cache.set(cache_key, reranked)
                
                return reranked[:top_k]
                
        except Exception as e:
            print(f"[Reranker] 重排序失败: {e}")
        
        # 失败时返回原始顺序
        return [
            RerankedDocument(
                document=doc,
                relevance_score=1.0 - min(doc.metadata.get("relevance_score", 0.5) / 2, 1.0),
                is_relevant=True
            )
            for doc in documents[:top_k]
        ]
    
    def check_scope(self, query: str) -> ScopeCheckResult:
        """
        使用LLM检测问题是否在刑法范围内（二次验证）
        
        Args:
            query: 用户问题
            
        Returns:
            超范围检测结果
        """
        # 检查缓存
        if self.cache:
            cache_key = self.cache._generate_key("scope_check", query)
            cached_result = self.cache.get(cache_key)
            if cached_result:
                return cached_result
        
        prompt = f"""你是法律领域分类专家。请判断以下问题属于哪个法律领域。

【用户问题】
{query}

【法律领域分类】
1. 刑法：犯罪、刑罚、刑事责任、故意杀人、盗窃、诈骗、强奸、走私等刑事犯罪
2. 民法：合同、婚姻、继承、物权、侵权、债务等民事关系
3. 商法：公司、证券、保险、票据、破产等商业事务
4. 行政法：行政处罚、行政复议、行政诉讼等行政管理
5. 劳动法：劳动合同、社保、工伤、劳动仲裁等劳动关系
6. 其他：税法、知识产权、环保等其他领域
7. 非法律：与法律无关的问题

【输出格式】（JSON）
{{
  "domain": "刑法/民法/商法/行政法/劳动法/其他/非法律",
  "confidence": 0.95,
  "reasoning": "简要说明判断理由"
}}

请只输出JSON。"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            result_text = response.content.strip()
            
            json_match = re.search(r'\{[\s\S]*\}', result_text)
            if json_match:
                result = json.loads(json_match.group())
                domain = result.get("domain", "刑法")
                confidence = result.get("confidence", 0.5)
                reasoning = result.get("reasoning", "")
                
                # 刑法领域视为在范围内
                is_in_scope = domain == "刑法"
                
                check_result = ScopeCheckResult(
                    is_in_scope=is_in_scope,
                    confidence=confidence,
                    detected_domain=domain,
                    reasoning=reasoning
                )
                
                # 缓存结果
                if self.cache:
                    self.cache.set(cache_key, check_result)
                
                return check_result
                
        except Exception as e:
            print(f"[Reranker] 范围检测失败: {e}")
        
        # 默认认为在范围内
        return ScopeCheckResult(
            is_in_scope=True,
            confidence=0.5,
            detected_domain="未知",
            reasoning="检测失败，默认在范围内"
        )
    
    def _check_hallucination_local(
        self,
        query: str,
        answer: str,
        retrieved_docs: List[Document]
    ) -> HallucinationCheckResult:
        """
        【方案B】本地快速幻觉检测 - 不调用LLM，使用启发式规则
        更快速，适合对正常问题的快速检测
        
        Args:
            query: 用户问题
            answer: 生成的回答
            retrieved_docs: 检索到的文档
            
        Returns:
            幻觉检测结果
        """
        problematic_claims = []
        risk_level = "low"
        
        # 提取检索文档中的关键信息
        context_text = " ".join([doc.page_content[:300] for doc in retrieved_docs[:3]])
        context_text_lower = context_text.lower()
        answer_lower = answer.lower()
        
        # 启发式规则1：检查法条编号
        # 提取答案中的法条编号（如"第232条"）
        law_articles = re.findall(r'第[0-9]+条', answer)
        for article in law_articles:
            if article not in context_text and len(context_text) > 100:
                # 答案中提到的法条不在检索结果中
                problematic_claims.append(f"法条 {article} 可能不在检索结果中")
        
        # 启发式规则2：检查量刑标准的一致性
        # 提取答案中的数字（年份、期数等）
        answer_numbers = re.findall(r'\d+年|死刑|无期|有期', answer)
        context_numbers = re.findall(r'\d+年|死刑|无期|有期', context_text)
        
        # 如果答案中有明显的量刑信息，检查是否与检索内容有重叠
        if answer_numbers and context_numbers:
            overlap = set(answer_numbers) & set(context_numbers)
            if not overlap and len(context_numbers) > 0:
                problematic_claims.append("量刑标准可能与检索内容不一致")
        
        # 启发式规则3：检查是否有明显的"创造性"信息
        # 查找以下模式：假设、如果、据说等不确定的表述
        uncertain_patterns = r'(据说|可能|听说|大概|好像|似乎|假设|如果)'
        if re.search(uncertain_patterns, answer):
            risk_level = "medium"
        
        # 启发式规则4：答案过长且检索文档很少，可能有填充
        if len(answer) > 1000 and len(retrieved_docs) < 2:
            problematic_claims.append("答案较长但检索文档较少，可能包含推断信息")
            risk_level = "medium"
        
        return HallucinationCheckResult(
            has_hallucination_risk=risk_level in ["medium", "high"],
            risk_level=risk_level,
            problematic_claims=problematic_claims,
            reasoning="基于启发式规则的本地检测"
        )
    
    def check_hallucination(
        self, 
        query: str, 
        answer: str, 
        retrieved_docs: List[Document],
        use_llm: bool = False
    ) -> HallucinationCheckResult:
        """
        检测回答中的潜在幻觉
        
        【方案B优化】支持选择本地快速检测或LLM深度检测
        
        Args:
            query: 用户问题
            answer: 生成的回答
            retrieved_docs: 检索到的文档
            use_llm: 是否使用LLM进行深度检测（比较慢）
            
        Returns:
            幻觉检测结果
        """
        if not ENABLE_HALLUCINATION_CHECK:
            return HallucinationCheckResult(
                has_hallucination_risk=False,
                risk_level="low",
                problematic_claims=[],
                reasoning="幻觉检测已禁用"
            )
        
        # 优先使用本地快速检测（默认）
        if not use_llm:
            return self._check_hallucination_local(query, answer, retrieved_docs)
        
        # 如果指定use_llm=True，使用LLM深度检测
        # 构建检索内容摘要
        context_summary = ""
        for i, doc in enumerate(retrieved_docs[:5]):
            content = doc.page_content[:200]
            context_summary += f"[来源{i+1}] {content}...\n"
        
        prompt = f"""你是法律事实核查专家。请检查以下回答是否存在幻觉（与检索内容不符的信息）。

【用户问题】
{query}

【检索到的法律文献】
{context_summary}

【生成的回答】
{answer[:800]}

【检查要求】
1. 检查回答中的法条编号是否与检索内容一致
2. 检查量刑标准是否与检索内容一致
3. 检查是否有检索内容中没有的"创造性"信息

【输出格式】（JSON）
{{
  "risk_level": "low/medium/high",
  "problematic_claims": ["问题陈述1", "问题陈述2"],
  "reasoning": "判断理由"
}}

请只输出JSON。"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            result_text = response.content.strip()
            
            json_match = re.search(r'\{[\s\S]*\}', result_text)
            if json_match:
                result = json.loads(json_match.group())
                risk_level = result.get("risk_level", "low")
                problematic_claims = result.get("problematic_claims", [])
                reasoning = result.get("reasoning", "")
                
                return HallucinationCheckResult(
                    has_hallucination_risk=risk_level in ["medium", "high"],
                    risk_level=risk_level,
                    problematic_claims=problematic_claims,
                    reasoning=reasoning
                )
                
        except Exception as e:
            print(f"[Reranker] LLM幻觉检测失败: {e}")
            # 回退到本地检测
            return self._check_hallucination_local(query, answer, retrieved_docs)
        
        return HallucinationCheckResult(
            has_hallucination_risk=False,
            risk_level="low",
            problematic_claims=[],
            reasoning="检测失败"
        )


# 单例模式
_reranker_instance: Optional[LLMReranker] = None

def get_reranker() -> LLMReranker:
    """获取重排序器单例"""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = LLMReranker()
    return _reranker_instance


# 便捷函数
def rerank_documents(query: str, documents: List[Document], top_k: int = None) -> List[RerankedDocument]:
    """便捷函数：重排序文档"""
    return get_reranker().rerank_documents(query, documents, top_k)


def check_scope(query: str) -> ScopeCheckResult:
    """便捷函数：检测范围"""
    return get_reranker().check_scope(query)


def check_hallucination(query: str, answer: str, docs: List[Document], use_llm: bool = False) -> HallucinationCheckResult:
    """便捷函数：检测幻觉
    
    Args:
        query: 用户问题
        answer: 生成的回答
        docs: 检索到的文档
        use_llm: 是否使用LLM进行深度检测（默认使用本地快速检测）
    """
    return get_reranker().check_hallucination(query, answer, docs, use_llm=use_llm)


if __name__ == "__main__":
    # 测试代码
    print("🧪 测试 Reranker 模块...")
    
    try:
        reranker = get_reranker()
        
        # 测试范围检测
        test_queries = [
            "故意杀人罪怎么判刑？",
            "离婚财产怎么分割？",
            "公司股权转让怎么办理？",
            "走私毒品判几年？"
        ]
        
        print("\n📌 范围检测测试：")
        for q in test_queries:
            result = reranker.check_scope(q)
            status = "✅" if result.is_in_scope else "❌"
            print(f"{status} [{result.detected_domain}] {q}")
            print(f"   置信度: {result.confidence:.2f}, 理由: {result.reasoning}")
        
        print("\n✅ Reranker 模块测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
