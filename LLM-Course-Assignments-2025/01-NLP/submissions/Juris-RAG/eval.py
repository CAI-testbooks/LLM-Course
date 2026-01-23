"""
Juris-RAG 评估模块
包含准确率、引用F1、幻觉率等评估指标
"""
import os
import json
import time
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict

# 导入配置和RAG引擎
try:
    from src.config import (
        EVAL_DATA_PATH, REPORTS_PATH, EVAL_BATCH_SIZE,
        SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL, LLM_MODEL,
        LLM_RPM_LIMIT, LLM_TPM_LIMIT, LLM_MIN_INTERVAL
    )
    from src.rag_engine import JurisRAGEngine, RAGResponse
except ImportError:
    EVAL_DATA_PATH = "./data/eval"
    REPORTS_PATH = "./reports"
    EVAL_BATCH_SIZE = 1
    SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
    SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
    LLM_MODEL = "Qwen/Qwen3-8B"
    LLM_RPM_LIMIT = int(os.getenv("LLM_RPM_LIMIT", "1000"))
    LLM_TPM_LIMIT = int(os.getenv("LLM_TPM_LIMIT", "50000"))
    LLM_MIN_INTERVAL = 60.0 / LLM_RPM_LIMIT if LLM_RPM_LIMIT > 0 else 0.0
    from rag_engine import JurisRAGEngine, RAGResponse


@dataclass
class EvalSample:
    """评估样本"""
    question: str
    ground_truth: str
    expected_sources: List[str] = None  # 期望引用的来源
    category: str = "general"  # 问题类别


@dataclass
class EvalResult:
    """单个样本的评估结果"""
    question: str
    ground_truth: str
    predicted_answer: str
    citations: List[str]
    confidence: float
    is_correct: bool
    citation_precision: float
    citation_recall: float
    citation_f1: float
    has_hallucination: bool
    relevance_score: float
    latency: float  # 响应时间（秒）


@dataclass
class EvalReport:
    """评估报告"""
    timestamp: str
    total_samples: int
    metrics: Dict
    category_metrics: Dict
    samples: List[EvalResult]


class JurisEvaluator:
    """法律RAG系统评估器"""
    
    def __init__(self):
        """初始化评估器"""
        self.engine = None
        self.eval_samples: List[EvalSample] = []
        self.results: List[EvalResult] = []
        self._last_call_ts: float = 0.0
        
    def initialize_engine(self):
        """初始化RAG引擎"""
        if self.engine is None:
            self.engine = JurisRAGEngine(streaming=False)
        return self.engine
    
    def load_eval_data(self, file_path: str = None) -> List[EvalSample]:
        """
        加载评估数据集
        
        Args:
            file_path: 评估数据文件路径（JSON Lines格式）
            
        Returns:
            List[EvalSample]: 评估样本列表
        """
        if file_path and os.path.exists(file_path):
            samples = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        sample = EvalSample(
                            question=data.get('question', ''),
                            ground_truth=data.get('answer', ''),
                            expected_sources=data.get('sources', []),
                            category=data.get('category', 'general')
                        )
                        samples.append(sample)
                    except:
                        continue
            self.eval_samples = samples
            print(f"✅ 加载了 {len(samples)} 个评估样本")
            return samples
        else:
            # 使用内置测试集
            print("📝 使用内置测试集...")
            self.eval_samples = self._get_builtin_eval_set()
            return self.eval_samples
    
    def _get_builtin_eval_set(self) -> List[EvalSample]:
        """获取内置的评估测试集"""
        return [
            # 刑法基础知识
            EvalSample(
                question="故意杀人罪怎么判刑？",
                ground_truth="故意杀人的，处死刑、无期徒刑或者十年以上有期徒刑；情节较轻的，处三年以上十年以下有期徒刑。",
                expected_sources=["刑法", "statute"],
                category="criminal_law"
            ),
            EvalSample(
                question="盗窃罪的量刑标准是什么？",
                ground_truth="盗窃公私财物，数额较大的，或者多次盗窃、入户盗窃、携带凶器盗窃、扒窃的，处三年以下有期徒刑、拘役或者管制，并处或者单处罚金。",
                expected_sources=["刑法", "statute"],
                category="criminal_law"
            ),
            EvalSample(
                question="什么是正当防卫？",
                ground_truth="为了使国家、公共利益、本人或者他人的人身、财产和其他权利免受正在进行的不法侵害，而采取的制止不法侵害的行为，对不法侵害人造成损害的，属于正当防卫，不负刑事责任。",
                expected_sources=["刑法", "statute"],
                category="criminal_law"
            ),
            EvalSample(
                question="抢劫罪怎么处罚？",
                ground_truth="以暴力、胁迫或者其他方法抢劫公私财物的，处三年以上十年以下有期徒刑，并处罚金。",
                expected_sources=["刑法", "statute"],
                category="criminal_law"
            ),
            EvalSample(
                question="故意伤害罪的刑期是多少？",
                ground_truth="故意伤害他人身体的，处三年以下有期徒刑、拘役或者管制。致人重伤的，处三年以上十年以下有期徒刑。",
                expected_sources=["刑法", "statute"],
                category="criminal_law"
            ),
            # 特殊情形
            EvalSample(
                question="未成年人犯罪怎么处理？",
                ground_truth="已满十四周岁不满十八周岁的人犯罪，应当从轻或者减轻处罚。",
                expected_sources=["刑法", "statute"],
                category="special_case"
            ),
            EvalSample(
                question="自首可以减刑吗？",
                ground_truth="犯罪以后自动投案，如实供述自己的罪行的，是自首。对于自首的犯罪分子，可以从轻或者减轻处罚。",
                expected_sources=["刑法", "statute"],
                category="special_case"
            ),
            # 案例相关
            EvalSample(
                question="诈骗案一般怎么判？",
                ground_truth="诈骗公私财物，数额较大的，处三年以下有期徒刑、拘役或者管制，并处或者单处罚金；数额巨大或者有其他严重情节的，处三年以上十年以下有期徒刑，并处罚金。",
                expected_sources=["刑法", "case"],
                category="case_related"
            ),
            EvalSample(
                question="交通肇事罪怎么判？",
                ground_truth="违反交通运输管理法规，因而发生重大事故，致人重伤、死亡或者使公私财产遭受重大损失的，处三年以下有期徒刑或者拘役。",
                expected_sources=["刑法", "statute"],
                category="criminal_law"
            ),
            # 扩展测试样本
            EvalSample(
                question="强奸罪如何判刑？",
                ground_truth="以暴力、胁迫或者其他手段强奸妇女的，处三年以上十年以下有期徒刑。",
                expected_sources=["刑法", "statute"],
                category="criminal_law"
            ),
            EvalSample(
                question="走私罪会面临什么处罚？",
                ground_truth="走私武器、弹药、核材料或者伪造的货币的，处七年以上有期徒刑，可以并处罚金或者没收财产。",
                expected_sources=["刑法", "statute"],
                category="criminal_law"
            ),
            EvalSample(
                question="贩毒罪的量刑标准？",
                ground_truth="走私、贩运、制造、持有、使用毒品的，依照本章规定处罚。量刑考虑毒品数量、性质等因素。",
                expected_sources=["刑法", "statute"],
                category="criminal_law"
            ),
            EvalSample(
                question="什么是共同犯罪？",
                ground_truth="二人以上共同故意犯罪的，是共同犯罪。共同犯罪人应当承担责任。",
                expected_sources=["刑法", "statute"],
                category="criminal_law"
            ),
            EvalSample(
                question="缓刑的条件是什么？",
                ground_truth="对被判处拘役、三年以下有期徒刑的犯罪分子，根据犯罪人的犯罪情节和悔罪表现，如果认为暂缓执行原判刑罚，确实不致再危害社会的，可以宣告缓刑。",
                expected_sources=["刑法", "statute"],
                category="special_case"
            ),
            EvalSample(
                question="累犯应该如何处罚？",
                ground_truth="前罪和后罪都是故意犯罪或者都是过失犯罪的，以及先后犯罪的间隔距离和犯罪人改造表现等情况，应当从重处罚。",
                expected_sources=["刑法", "statute"],
                category="special_case"
            ),
            EvalSample(
                question="贪污贿赂罪怎么判？",
                ground_truth="国家工作人员利用职务上的便利，索取他人财物或者非法收受他人财物，为他人谋取利益的，是受贿罪。",
                expected_sources=["刑法", "statute"],
                category="criminal_law"
            ),
            EvalSample(
                question="伪证罪的定义是什么？",
                ground_truth="在刑事诉讼中，证人、鉴定人、记录人、翻译人故意作虚假证明、鉴定、记录、翻译，意图陷害他人或者隐匿罪证的，处三年以下有期徒刑或者拘役。",
                expected_sources=["刑法", "statute"],
                category="criminal_law"
            ),
            EvalSample(
                question="什么时候应该减轻处罚？",
                ground_truth="犯罪情节轻微、危害不大的，不认为是犯罪，或者可以减轻处罚。",
                expected_sources=["刑法", "statute"],
                category="special_case"
            ),
            EvalSample(
                question="抢劫罪和盗窃罪的区别？",
                ground_truth="抢劫罪以暴力、胁迫或者其他方法强行夺取财物；盗窃罪是秘密盗取。二者手段完全不同。",
                expected_sources=["刑法", "statute"],
                category="criminal_law"
            ),
            EvalSample(
                question="聚众斗殴罪如何处罚？",
                ground_truth="聚众斗殴的，对首要分子和其他积极参加的，处三年以下有期徒刑、拘役或者管制。",
                expected_sources=["刑法", "statute"],
                category="criminal_law"
            ),
            EvalSample(
                question="包庇罪的构成条件？",
                ground_truth="明知是犯罪的人而为其隐瞒、包庇，或者为其提供逃匿的便利，处三年以下有期徒刑、拘役或者管制。",
                expected_sources=["刑法", "statute"],
                category="criminal_law"
            ),
            EvalSample(
                question="妨害公务罪怎么处罚？",
                ground_truth="以暴力、威胁方法阻碍国家机关工作人员依法执行职务的，处三年以下有期徒刑、拘役、管制或罚金。",
                expected_sources=["刑法", "statute"],
                category="criminal_law"
            ),
            # 边界测试
            EvalSample(
                question="民法典关于合同的规定是什么？",
                ground_truth="",  # 超出范围，应该拒绝回答
                expected_sources=[],
                category="out_of_scope"
            ),
            EvalSample(
                question="股票投资有什么法律风险？",
                ground_truth="",  # 超出范围，应该拒绝回答
                expected_sources=[],
                category="out_of_scope"
            ),
            EvalSample(
                question="公司法中关于董事责任的规定是什么？",
                ground_truth="",  # 超出范围，应该拒绝回答
                expected_sources=[],
                category="out_of_scope"
            ),
        ]
    
    def evaluate_single(self, sample: EvalSample) -> EvalResult:
        """
        评估单个样本
        
        Args:
            sample: 评估样本
            
        Returns:
            EvalResult: 评估结果
        """
        start_time = time.time()

        # 简单速率限制（按 RPM 节流）
        self._respect_rate_limit()
        
        # 获取模型回答
        response = self.engine.query(sample.question)
        
        latency = time.time() - start_time
        
        # 提取引用来源
        citations = [c.source for c in response.citations]
        
        # 计算各项指标
        is_correct = self._check_correctness(
            response.answer, 
            sample.ground_truth, 
            sample.category
        )
        
        citation_precision, citation_recall, citation_f1 = self._calculate_citation_f1(
            citations,
            sample.expected_sources or []
        )
        
        has_hallucination = self._detect_hallucination(
            response.answer,
            response.retrieved_docs,
            sample.ground_truth
        )
        
        relevance_score = self._calculate_relevance(
            response.answer,
            sample.question,
            sample.ground_truth
        )
        
        # 清空对话历史，避免影响下一个样本
        self.engine.clear_history()
        
        return EvalResult(
            question=sample.question,
            ground_truth=sample.ground_truth,
            predicted_answer=response.answer,
            citations=citations,
            confidence=response.confidence,
            is_correct=is_correct,
            citation_precision=citation_precision,
            citation_recall=citation_recall,
            citation_f1=citation_f1,
            has_hallucination=has_hallucination,
            relevance_score=relevance_score,
            latency=latency
        )

    def _respect_rate_limit(self):
        """基于 RPM 做最小间隔节流，适配 L0 配额。"""
        if LLM_MIN_INTERVAL <= 0:
            return
        now = time.time()
        wait = LLM_MIN_INTERVAL - (now - self._last_call_ts)
        if wait > 0:
            time.sleep(wait)
        self._last_call_ts = time.time()
    
    def _check_correctness(
        self, 
        predicted: str, 
        ground_truth: str,
        category: str
    ) -> bool:
        """
        检查回答是否正确
        更严格的评估标准：内容必须基本一致
        """
        if category == "out_of_scope":
            # 对于超出范围的问题，拒绝回答视为正确
            refuse_keywords = ["无法回答", "不在", "超出", "无法准确", "没有相关", "无法找到", "超出"]
            return any(kw in predicted for kw in refuse_keywords)
        
        if not ground_truth:
            return False
        
        # 提取关键要素进行匹配
        
        # 1. 提取刑期数字（最重要）
        gt_numbers = re.findall(r'(\d+)年', ground_truth)
        pred_numbers = re.findall(r'(\d+)年', predicted)
        
        # 2. 检查关键刑罚词
        penalty_keywords = ['死刑', '无期徒刑', '有期徒刑', '拘役', '管制', '罚金', '没收']
        gt_penalties = set(kw for kw in penalty_keywords if kw in ground_truth)
        pred_penalties = set(kw for kw in penalty_keywords if kw in predicted)
        
        # 3. 检查是否拒绝回答（如果应该回答却拒绝了）
        refused = "无法回答" in predicted or "无法准确" in predicted or "无法找到" in predicted
        
        if refused and ground_truth:
            # 应该能回答但拒绝了 -> 错误
            return False
        
        # 4. 综合判断正确性
        # 严格标准：刑期和刑罚词都要匹配
        
        if gt_penalties:
            # 有刑罚词的，刑罚词匹配度要高
            penalty_match_ratio = len(gt_penalties & pred_penalties) / len(gt_penalties)
            if penalty_match_ratio < 0.5:
                return False  # 主要刑罚词都没对上
        
        if gt_numbers:
            # 有数字的（刑期），数字匹配度也要高
            number_match_ratio = len(set(gt_numbers) & set(pred_numbers)) / len(gt_numbers)
            if number_match_ratio < 0.5:
                # 主要的刑期数字都没对上
                return False
        
        # 检查关键法律概念是否包含
        if "条" in ground_truth:
            # 如果是法条定义，至少要有法律概念的匹配
            concept_words = ['定义', '规定', '是指', '处', '处罚', '刑事责任']
            gt_has_concept = any(w in ground_truth for w in concept_words)
            pred_has_concept = any(w in predicted for w in concept_words)
            
            if gt_has_concept and not pred_has_concept:
                return False
        
        # 如果主要要素都匹配了，视为正确
        return True
    
    def _calculate_citation_f1(
        self, 
        predicted_sources: List[str],
        expected_sources: List[str]
    ) -> Tuple[float, float, float]:
        """
        计算引用F1值（改进版）
        
        Returns:
            Tuple[precision, recall, f1]
        """
        if not expected_sources:
            # 没有期望来源时，有引用就算正确
            return (1.0, 1.0, 1.0) if predicted_sources else (0.0, 0.0, 0.0)
        
        if not predicted_sources:
            return (0.0, 0.0, 0.0)
        
        # 将来源标准化并进行部分匹配
        def normalize_source(s):
            s = s.lower().strip()
            # 移除常见噪音
            for noise in ['（', '）', '【', '】', '(', ')']:
                s = s.replace(noise, ' ')
            return s
        
        # 定义匹配关键词
        source_keywords = {
            'statute': ['刑法', 'statute', '法条', '条'],
            'case': ['cail', '案例', 'case', '司法']
        }
        
        # 检查预测来源是否匹配期望来源
        matches = 0
        for exp in expected_sources:
            exp_norm = normalize_source(exp)
            for pred in predicted_sources:
                pred_norm = normalize_source(pred)
                # 直接匹配
                if exp_norm in pred_norm or pred_norm in exp_norm:
                    matches += 1
                    break
                # 关键词匹配
                for category, keywords in source_keywords.items():
                    if any(kw in exp_norm for kw in keywords) and any(kw in pred_norm for kw in keywords):
                        matches += 1
                        break
        
        precision = min(matches / len(predicted_sources), 1.0) if predicted_sources else 0
        recall = min(matches / len(expected_sources), 1.0) if expected_sources else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return (precision, recall, f1)
    
    def _detect_hallucination(
        self,
        answer: str,
        retrieved_docs: List,
        ground_truth: str
    ) -> bool:
        """
        检测幻觉 - 平衡检测
        只检测明确的编造行为，避免过于严格导致误判
        """
        # 如果回答明确表示无法回答，不算幻觉
        refusal_keywords = ["无法回答", "无法准确", "未找到相关", "检索内容中未找到", "没有相关"]
        if any(kw in answer for kw in refusal_keywords):
            return False
        
        # 如果没有检索到文档但有实质性答案，可能是幻觉
        if not retrieved_docs and answer and len(answer) > 100:
            return True
        
        if not retrieved_docs:
            return False
        
        # 提取回答中的法条编号
        answer_articles = set(re.findall(r'第[一二三四五六七八九十百千零\d]+条', answer))
        
        # 如果回答没有引用具体法条，不检测幻觉（可能是概括性回答）
        if not answer_articles:
            return False
        
        # 提取检索文档中的法条编号
        doc_articles = set()
        doc_full_text = ""
        for doc in retrieved_docs:
            doc_articles.update(re.findall(r'第[一二三四五六七八九十百千零\d]+条', doc.page_content))
            doc_full_text += doc.page_content
        
        # 允许ground_truth中的法条
        if ground_truth:
            gt_articles = set(re.findall(r'第[一二三四五六七八九十百千零\d]+条', ground_truth))
            doc_articles = doc_articles | gt_articles
        
        # 检测：回答中超过半数的法条在文档中找不到
        hallucinated_articles = answer_articles - doc_articles
        if len(hallucinated_articles) > len(answer_articles) / 2:
            return True
        
        return False
    
    def _calculate_relevance(
        self,
        answer: str,
        question: str,
        ground_truth: str
    ) -> float:
        """
        计算回答相关性分数
        简单实现：基于关键词重叠度
        """
        import jieba
        
        # 分词
        q_words = set(jieba.cut(question))
        a_words = set(jieba.cut(answer))
        gt_words = set(jieba.cut(ground_truth)) if ground_truth else set()
        
        # 过滤停用词
        stopwords = {'的', '是', '在', '了', '和', '与', '或', '等', '有', '对', '被', '为', '以', '及'}
        q_words = q_words - stopwords
        a_words = a_words - stopwords
        gt_words = gt_words - stopwords
        
        # 计算与问题的相关性
        q_overlap = len(q_words & a_words) / max(len(q_words), 1)
        
        # 计算与标准答案的相关性
        if gt_words:
            gt_overlap = len(gt_words & a_words) / max(len(gt_words), 1)
        else:
            gt_overlap = 0.5  # 没有标准答案时给中等分
        
        # 综合得分
        relevance = 0.4 * q_overlap + 0.6 * gt_overlap
        return min(relevance, 1.0)
    
    def run_evaluation(self, samples: List[EvalSample] = None) -> EvalReport:
        """
        运行完整评估
        
        Args:
            samples: 评估样本列表，为空则使用已加载的样本
            
        Returns:
            EvalReport: 评估报告
        """
        if samples is None:
            samples = self.eval_samples
        
        if not samples:
            print("❌ 没有评估样本！请先加载数据。")
            return None
        
        print(f"\n🧪 开始评估，共 {len(samples)} 个样本...")
        print("=" * 60)
        
        # 初始化引擎
        self.initialize_engine()
        
        self.results = []
        category_results = defaultdict(list)
        
        for i, sample in enumerate(samples, 1):
            print(f"\n[{i}/{len(samples)}] 评估: {sample.question[:30]}...")
            
            attempt = 0
            max_retries = 2
            backoff = 5
            while attempt <= max_retries:
                try:
                    result = self.evaluate_single(sample)
                    self.results.append(result)
                    category_results[sample.category].append(result)
                    
                    status = "✅" if result.is_correct else "❌"
                    print(f"   {status} 正确性: {result.is_correct}, 置信度: {result.confidence:.2f}, 耗时: {result.latency:.2f}s")
                    break
                except Exception as e:
                    if self._is_rate_limit_error(e) and attempt < max_retries:
                        wait_seconds = backoff * (2 ** attempt)
                        print(f"   ⚠️ 触发限流，等待 {wait_seconds}s 后重试 (第 {attempt+1}/{max_retries} 次)")
                        time.sleep(wait_seconds)
                        attempt += 1
                        continue
                    print(f"   ⚠️ 评估失败: {e}")
                    break
            
            # 批次间隔
            if i % EVAL_BATCH_SIZE == 0:
                print(f"\n   已完成 {i}/{len(samples)} ({100*i/len(samples):.1f}%)")
                # 额外节流：按照 RPM 追加等待，避免短时间过多请求
                if LLM_MIN_INTERVAL > 0:
                    time.sleep(LLM_MIN_INTERVAL)
        
        # 计算总体指标
        metrics = self._calculate_metrics(self.results)
        
        # 计算分类指标
        category_metrics = {}
        for category, results in category_results.items():
            category_metrics[category] = self._calculate_metrics(results)
        
        # 生成报告
        report = EvalReport(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_samples=len(samples),
            metrics=metrics,
            category_metrics=category_metrics,
            samples=self.results
        )
        
        return report

    @staticmethod
    def _is_rate_limit_error(err: Exception) -> bool:
        msg = str(err).lower()
        return any(key in msg for key in ["rpm limit", "rate limit", "429", "too many", "exceeded"])
    
    def _calculate_metrics(self, results: List[EvalResult]) -> Dict:
        """计算评估指标"""
        if not results:
            return {}
        
        n = len(results)
        
        return {
            "accuracy": sum(1 for r in results if r.is_correct) / n,
            "avg_confidence": sum(r.confidence for r in results) / n,
            "citation_precision": sum(r.citation_precision for r in results) / n,
            "citation_recall": sum(r.citation_recall for r in results) / n,
            "citation_f1": sum(r.citation_f1 for r in results) / n,
            "hallucination_rate": sum(1 for r in results if r.has_hallucination) / n,
            "avg_relevance": sum(r.relevance_score for r in results) / n,
            "avg_latency": sum(r.latency for r in results) / n,
            "total_samples": n
        }
    
    def print_report(self, report: EvalReport):
        """打印评估报告"""
        print("\n" + "=" * 60)
        print("📊 评估报告")
        print("=" * 60)
        print(f"⏰ 评估时间: {report.timestamp}")
        print(f"📝 样本数量: {report.total_samples}")

        if not report.metrics:
            print("⚠️ 无可用评估结果（全部样本失败或被跳过）")
            print("=" * 60)
            return

        print("\n📈 总体指标:")
        print("-" * 40)
        m = report.metrics
        print(f"   准确率 (Accuracy):     {m['accuracy']:.2%}")
        print(f"   平均置信度:            {m['avg_confidence']:.2%}")
        print(f"   引用精确率:            {m['citation_precision']:.2%}")
        print(f"   引用召回率:            {m['citation_recall']:.2%}")
        print(f"   引用F1:               {m['citation_f1']:.2%}")
        print(f"   幻觉率:               {m['hallucination_rate']:.2%}")
        print(f"   平均相关性:            {m['avg_relevance']:.2%}")
        print(f"   平均响应时间:          {m['avg_latency']:.2f}s")
        
        if report.category_metrics:
            print("\n📂 分类指标:")
            print("-" * 40)
            for category, metrics in report.category_metrics.items():
                print(f"\n   【{category}】(n={metrics['total_samples']})")
                print(f"      准确率: {metrics['accuracy']:.2%}")
                print(f"      引用F1: {metrics['citation_f1']:.2%}")
                print(f"      幻觉率: {metrics['hallucination_rate']:.2%}")
        
        print("\n" + "=" * 60)
    
    def save_report(self, report: EvalReport, output_path: str = None):
        """保存评估报告"""
        if output_path is None:
            os.makedirs(REPORTS_PATH, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(REPORTS_PATH, f"eval_report_{timestamp}.json")
        
        # 转换为可序列化格式
        report_dict = {
            "timestamp": report.timestamp,
            "total_samples": report.total_samples,
            "metrics": report.metrics,
            "category_metrics": report.category_metrics,
            "samples": [asdict(s) for s in report.samples]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, ensure_ascii=False, indent=2)
        
        print(f"📄 报告已保存至: {output_path}")
        return output_path


def run_baseline_comparison():
    """
    运行基线对比实验
    比较不同配置下的性能
    """
    print("🔬 基线对比实验")
    print("=" * 60)
    
    evaluator = JurisEvaluator()
    evaluator.load_eval_data()
    
    # 运行评估
    report = evaluator.run_evaluation()
    
    if report:
        evaluator.print_report(report)
        evaluator.save_report(report)
    
    return report


def main():
    """主函数"""
    print("🏛️ Juris-RAG 评估系统")
    print("=" * 60)
    
    # 检查环境
    if not SILICONFLOW_API_KEY:
        print("❌ 请先设置 SILICONFLOW_API_KEY 环境变量！")
        return
    
    # 创建评估器
    evaluator = JurisEvaluator()
    
    # 加载评估数据
    eval_file = os.path.join(EVAL_DATA_PATH, "eval_set.json")
    evaluator.load_eval_data(eval_file)
    
    # 运行评估
    report = evaluator.run_evaluation()
    
    if report:
        # 打印报告
        evaluator.print_report(report)
        
        # 保存报告
        evaluator.save_report(report)
        
        print("\n✅ 评估完成！")


if __name__ == "__main__":
    main()
