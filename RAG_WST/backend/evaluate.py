"""
RAG 评测脚本
支持自定义数据集评测，使用 LLM 判断答案正确性
评测指标：准确率、引用F1、幻觉率、Hit@K、MRR
"""

import os
import sys
import json
import re
import csv
import time
import random
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from tqdm import tqdm
import numpy as np

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag import RAG


# ==================== 数据结构 ====================

@dataclass
class Question:
    """问题"""
    question_id: str
    content: str


@dataclass
class Answer:
    """答案"""
    ans_id: str
    question_id: str
    content: str


@dataclass
class EvalSample:
    """评测样本"""
    question_id: str
    question_content: str
    correct_ans_ids: List[str]  # 正确答案ID列表
    correct_answers: List[str]   # 正确答案内容列表


@dataclass
class EvalResult:
    """单条评测结果"""
    question_id: str
    question: str
    ground_truths: List[str]      # 标准答案列表
    prediction: str               # 模型生成的答案
    retrieved_docs: List[dict]    # 检索到的文档
    retrieved_ans_ids: List[str]  # 检索到的答案ID
    
    # 检索指标
    hit: bool = False             # 是否命中正确答案
    hit_rank: int = -1            # 命中排名（-1表示未命中）
    
    # 生成指标（由LLM评判）
    is_correct: bool = False      # 答案是否正确
    citation_precision: float = 0.0  # 引用精确率
    citation_recall: float = 0.0     # 引用召回率
    hallucination_rate: float = 0.0  # 幻觉率
    
    # LLM 评判原始结果
    llm_judgment: dict = field(default_factory=dict)


@dataclass
class EvalMetrics:
    """评测指标汇总"""
    total_samples: int = 0
    
    # 检索指标
    hit_at_1: float = 0.0
    hit_at_3: float = 0.0
    hit_at_5: float = 0.0
    hit_at_10: float = 0.0
    mrr: float = 0.0
    
    # 生成指标
    accuracy: float = 0.0
    citation_precision: float = 0.0
    citation_recall: float = 0.0
    citation_f1: float = 0.0
    hallucination_rate: float = 0.0
    
    # 耗时
    elapsed_seconds: float = 0.0


# ==================== 数据加载 ====================

class DataLoader:
    """数据加载器"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.questions: Dict[str, Question] = {}
        self.answers: Dict[str, Answer] = {}
    
    def _detect_delimiter(self, file_path: Path) -> str:
        """检测分隔符"""
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if '\t' in first_line:
                return '\t'
            return ','
    
    def load_questions(self, filename: str = "question.csv") -> Dict[str, Question]:
        """加载问题"""
        file_path = self.data_dir / filename
        delimiter = self._detect_delimiter(file_path)
        
        print(f"Loading questions from {file_path} (delimiter: {'tab' if delimiter == '\t' else 'comma'})")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                qid = str(row['question_id']).strip()
                content = row['content'].strip()
                self.questions[qid] = Question(question_id=qid, content=content)
        
        print(f"  Loaded {len(self.questions)} questions")
        return self.questions
    
    def load_answers(self, filename: str = "answer.csv") -> Dict[str, Answer]:
        """加载答案"""
        file_path = self.data_dir / filename
        delimiter = self._detect_delimiter(file_path)
        
        print(f"Loading answers from {file_path} (delimiter: {'tab' if delimiter == '\t' else 'comma'})")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                ans_id = str(row['ans_id']).strip()
                qid = str(row['question_id']).strip()
                content = row['content'].strip()
                self.answers[ans_id] = Answer(ans_id=ans_id, question_id=qid, content=content)
        
        print(f"  Loaded {len(self.answers)} answers")
        return self.answers
    
    def load_train_candidates(self, filename: str = "train_candidates.txt") -> List[Tuple[str, str]]:
        """
        加载训练集候选（用于构建知识库）
        返回：[(question_id, pos_ans_id), ...]
        """
        file_path = self.data_dir / filename
        
        print(f"Loading train candidates from {file_path}")
        
        pairs = set()  # 去重
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                qid = str(row['question_id']).strip()
                pos_ans_id = str(row['pos_ans_id']).strip()
                pairs.add((qid, pos_ans_id))
        
        result = list(pairs)
        print(f"  Loaded {len(result)} unique (question, answer) pairs")
        return result
    
    def load_eval_candidates(self, filename: str) -> List[EvalSample]:
        """
        加载评测集候选（dev 或 test）
        返回：EvalSample 列表
        """
        file_path = self.data_dir / filename
        
        print(f"Loading eval candidates from {file_path}")
        
        # 按问题ID分组，收集正确答案
        question_answers: Dict[str, List[str]] = defaultdict(list)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                qid = str(row['question_id']).strip()
                ans_id = str(row['ans_id']).strip()
                label = int(row['label'])
                
                if label == 1:
                    question_answers[qid].append(ans_id)
        
        # 构建评测样本
        samples = []
        for qid, ans_ids in question_answers.items():
            if qid not in self.questions:
                print(f"  Warning: question {qid} not found, skipping")
                continue
            
            question = self.questions[qid]
            correct_answers = []
            for ans_id in ans_ids:
                if ans_id in self.answers:
                    correct_answers.append(self.answers[ans_id].content)
                else:
                    print(f"  Warning: answer {ans_id} not found")
            
            if correct_answers:
                samples.append(EvalSample(
                    question_id=qid,
                    question_content=question.content,
                    correct_ans_ids=ans_ids,
                    correct_answers=correct_answers
                ))
        
        print(f"  Loaded {len(samples)} eval samples")
        return samples


# ==================== LLM 评判器 ====================

class LLMJudge:
    """使用 LLM 评判答案质量"""
    
    JUDGE_PROMPT = """你是一个专业的医学问答评测专家。请评估AI助手的回答质量。

## 评测任务

给定：
1. 用户问题
2. 标准答案（可能有多个）
3. AI助手的回答
4. AI引用的参考来源

请评估以下指标：

### 1. 正确性 (is_correct)
AI的回答是否正确回答了用户问题？与标准答案的核心信息是否一致？
- true: 回答正确，核心信息与标准答案一致
- false: 回答错误或偏离主题

### 2. 引用相关性 (citation_relevance)
AI引用的来源中，有多少是真正支持其回答的？
- 返回一个 0-1 之间的分数
- 1.0 = 所有引用都相关
- 0.0 = 没有引用或所有引用都不相关

### 3. 引用覆盖度 (citation_coverage)  
AI的回答中，有多少关键信息有引用支持？
- 返回一个 0-1 之间的分数
- 1.0 = 所有关键信息都有引用
- 0.0 = 没有任何引用支持

### 4. 幻觉率 (hallucination_rate)
AI的回答中，有多少信息是编造的（不在引用来源中，也不是常识）？
- 返回一个 0-1 之间的分数
- 0.0 = 没有幻觉
- 1.0 = 完全是幻觉

## 输入信息

**用户问题：**
{question}

**标准答案：**
{ground_truths}

**AI回答：**
{prediction}

**AI引用的来源：**
{sources}

## 输出格式

请严格按照以下 JSON 格式输出，不要输出其他内容：

```json
{{
    "is_correct": true/false,
    "citation_relevance": 0.0-1.0,
    "citation_coverage": 0.0-1.0,
    "hallucination_rate": 0.0-1.0,
    "reasoning": "简要说明评判理由"
}}
```"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def judge(
        self,
        question: str,
        ground_truths: List[str],
        prediction: str,
        sources: List[dict]
    ) -> dict:
        """评判单条结果"""
        
        # 格式化标准答案
        gt_text = "\n".join([f"答案{i+1}: {gt}" for i, gt in enumerate(ground_truths)])
        
        # 格式化引用来源
        if sources:
            sources_text = "\n".join([
                f"[来源{s.get('index', i+1)}] {s.get('content', '')[:500]}"
                for i, s in enumerate(sources)
            ])
        else:
            sources_text = "（无引用来源）"
        
        prompt = self.JUDGE_PROMPT.format(
            question=question,
            ground_truths=gt_text,
            prediction=prediction,
            sources=sources_text
        )
        
        try:
            response = self.llm.generate(
                prompt,
                system_prompt="你是一个严格的评测专家，请按照要求输出 JSON 格式的评测结果。",
                temperature=0.1,
                max_tokens=512
            )
            
            # 提取 JSON
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    "is_correct": bool(result.get("is_correct", False)),
                    "citation_relevance": float(result.get("citation_relevance", 0.0)),
                    "citation_coverage": float(result.get("citation_coverage", 0.0)),
                    "hallucination_rate": float(result.get("hallucination_rate", 0.0)),
                    "reasoning": result.get("reasoning", "")
                }
        except Exception as e:
            print(f"  LLM judge error: {e}")
        
        # 默认返回
        return {
            "is_correct": False,
            "citation_relevance": 0.0,
            "citation_coverage": 0.0,
            "hallucination_rate": 1.0,
            "reasoning": "评判失败"
        }


# ==================== 评测器 ====================

class RAGEvaluator:
    """RAG 评测器"""
    
    def __init__(
        self,
        rag: RAG,
        data_dir: str,
        use_llm_judge: bool = True,
        seed: int = 42
    ):
        self.rag = rag
        self.data_loader = DataLoader(data_dir)
        self.use_llm_judge = use_llm_judge
        self.judge = LLMJudge(rag.llm) if use_llm_judge else None
        
        # 用于检查数据重叠
        self.train_question_ids: set = set()
        self.train_ans_ids: set = set()
        
        random.seed(seed)
        np.random.seed(seed)
    
    def _check_data_overlap(self, eval_samples: List[EvalSample]):
        """检查训练数据和评测数据的重叠情况"""
        print(f"\n[DEBUG] Data Overlap Check:")
        print(f"  Train question_ids: {len(self.train_question_ids)}")
        print(f"  Train ans_ids: {len(self.train_ans_ids)}")
        
        eval_question_ids = set(s.question_id for s in eval_samples)
        eval_ans_ids = set()
        for s in eval_samples:
            eval_ans_ids.update(s.correct_ans_ids)
        
        print(f"  Eval question_ids: {len(eval_question_ids)}")
        print(f"  Eval ans_ids: {len(eval_ans_ids)}")
        
        # 检查重叠
        qid_overlap = self.train_question_ids & eval_question_ids
        ans_overlap = self.train_ans_ids & eval_ans_ids
        
        print(f"\n  Question ID overlap: {len(qid_overlap)} ({len(qid_overlap)/len(eval_question_ids)*100:.1f}%)")
        print(f"  Answer ID overlap: {len(ans_overlap)} ({len(ans_overlap)/len(eval_ans_ids)*100:.1f}%)")
        
        if len(ans_overlap) == 0:
            print(f"\n  ⚠️ WARNING: No answer ID overlap between train and eval data!")
            print(f"     This means retrieval Hit@K metrics will always be 0.")
            print(f"     The evaluation will fallback to question_id matching.")
        
        # 打印一些样例
        print(f"\n  Sample train ans_ids: {list(self.train_ans_ids)[:5]}")
        print(f"  Sample eval ans_ids: {list(eval_ans_ids)[:5]}")
    
    def build_knowledge_base(self, max_docs: int = None, batch_size: int = 64) -> int:
        """
        构建知识库
        
        Args:
            max_docs: 最大文档数
            batch_size: 批量 embedding 大小
        """
        print(f"\n{'='*50}")
        print("Building Knowledge Base")
        print(f"{'='*50}")
        
        # 加载数据
        self.data_loader.load_questions()
        self.data_loader.load_answers()
        train_pairs = self.data_loader.load_train_candidates()
        
        # 存储训练数据的 IDs（用于后续检查重叠）
        for qid, ans_id in train_pairs:
            self.train_question_ids.add(qid)
            self.train_ans_ids.add(ans_id)
        
        if max_docs and len(train_pairs) > max_docs:
            print(f"  Limiting to {max_docs} documents")
            random.shuffle(train_pairs)
            train_pairs = train_pairs[:max_docs]
        
        # 清空已有知识库
        self.rag.vector_store.clear()
        
        total_chunks = 0
        skipped = 0
        batch_count = 0
        
        # 使用批量处理
        chunk_buffer = []
        
        for qid, ans_id in tqdm(train_pairs, desc="Indexing"):
            # 获取问题和答案
            if qid not in self.data_loader.questions:
                skipped += 1
                continue
            if ans_id not in self.data_loader.answers:
                skipped += 1
                continue
            
            question = self.data_loader.questions[qid]
            answer = self.data_loader.answers[ans_id]
            
            # 构建文档：问题 + 答案
            doc_content = f"问题：{question.content}\n答案：{answer.content}"
            doc_name = f"qa_{qid}_{ans_id}"
            
            # 分块
            chunks = self.rag.chunker.chunk_text(doc_content, doc_name)
            if not chunks:
                skipped += 1
                continue
            
            # 存储元数据
            for chunk in chunks:
                chunk.metadata = {
                    "question_id": qid,
                    "ans_id": ans_id,
                    "question": question.content,
                    "answer": answer.content
                }
            
            # 添加到缓冲区
            chunk_buffer.extend(chunks)
            total_chunks += len(chunks)
            
            # 达到批量大小时进行 embedding
            if len(chunk_buffer) >= batch_size:
                embeddings = self.rag.embedder.embed([c.content for c in chunk_buffer])
                self.rag.vector_store.add(chunk_buffer, embeddings)
                batch_count += 1
                chunk_buffer = []
        
        # 处理剩余的 chunks
        if chunk_buffer:
            embeddings = self.rag.embedder.embed([c.content for c in chunk_buffer])
            self.rag.vector_store.add(chunk_buffer, embeddings)
            batch_count += 1
        
        # 保存索引到磁盘（只保存一次！）
        self.rag.vector_store.save()
        
        print(f"\nKnowledge base built:")
        print(f"  Total chunks: {total_chunks}")
        print(f"  Batch count: {batch_count} (batch_size={batch_size})")
        print(f"  Skipped: {skipped}")
        
        return total_chunks
    
    def _evaluate_single(
        self,
        sample: EvalSample,
        top_k: int = 5,
        debug: bool = False
    ) -> EvalResult:
        """评测单个样本（供并发调用）"""
        # 1. 检索
        retrieval_results = self.rag.retrieve(sample.question_content, top_k=max(top_k, 10))
        
        # 提取检索到的答案ID
        retrieved_ans_ids = []
        retrieved_question_ids = []
        retrieved_docs = []
        for i, r in enumerate(retrieval_results):
            ans_id = r.chunk.metadata.get("ans_id", "")
            question_id = r.chunk.metadata.get("question_id", "")
            retrieved_ans_ids.append(ans_id)
            retrieved_question_ids.append(question_id)
            retrieved_docs.append({
                "index": i + 1,
                "doc_name": r.chunk.doc_name,
                "score": round(r.score, 4),
                "content": r.chunk.content,
                "ans_id": ans_id,
                "question_id": question_id
            })
        
        # 2. 计算检索指标 - 支持按 ans_id 或 question_id 匹配
        hit = False
        hit_rank = -1
        
        # 方式1: 按 ans_id 匹配
        for i, ans_id in enumerate(retrieved_ans_ids):
            if ans_id and ans_id in sample.correct_ans_ids:
                hit = True
                hit_rank = i + 1
                break
        
        # 方式2: 如果 ans_id 没命中，尝试按 question_id 匹配
        if not hit:
            for i, qid in enumerate(retrieved_question_ids):
                if qid and qid == sample.question_id:
                    hit = True
                    hit_rank = i + 1
                    break
        
        # 调试信息
        if debug and not hit:
            print(f"\n[DEBUG] Question {sample.question_id}:")
            print(f"  Correct ans_ids: {sample.correct_ans_ids}")
            print(f"  Retrieved ans_ids: {retrieved_ans_ids[:5]}")
            print(f"  Retrieved question_ids: {retrieved_question_ids[:5]}")
        
        # 3. 生成回答
        answer = ""
        sources = []
        try:
            answer, sources = self.rag.query(sample.question_content, top_k=top_k)
        except Exception as e:
            pass  # 静默处理错误
        
        # 4. LLM 评判
        llm_judgment = {}
        is_correct = False
        citation_precision = 0.0
        citation_recall = 0.0
        hallucination_rate = 0.0
        
        if self.use_llm_judge and self.judge and answer:
            try:
                llm_judgment = self.judge.judge(
                    question=sample.question_content,
                    ground_truths=sample.correct_answers,
                    prediction=answer,
                    sources=sources
                )
                
                is_correct = llm_judgment.get("is_correct", False)
                citation_precision = llm_judgment.get("citation_relevance", 0.0)
                citation_recall = llm_judgment.get("citation_coverage", 0.0)
                hallucination_rate = llm_judgment.get("hallucination_rate", 0.0)
            except Exception as e:
                pass  # 静默处理错误
        
        # 5. 返回结果
        return EvalResult(
            question_id=sample.question_id,
            question=sample.question_content,
            ground_truths=sample.correct_answers,
            prediction=answer,
            retrieved_docs=retrieved_docs,
            retrieved_ans_ids=retrieved_ans_ids,
            hit=hit,
            hit_rank=hit_rank,
            is_correct=is_correct,
            citation_precision=citation_precision,
            citation_recall=citation_recall,
            hallucination_rate=hallucination_rate,
            llm_judgment=llm_judgment
        )
    
    def evaluate(
        self,
        eval_file: str = "dev_candidates.txt",
        max_samples: int = None,
        top_k: int = 5,
        workers: int = 1,
        debug: bool = False
    ) -> Tuple[EvalMetrics, List[EvalResult]]:
        """
        执行评测
        
        Args:
            eval_file: 评测文件
            max_samples: 最大样本数
            top_k: 检索 Top-K
            workers: 并发线程数（默认 1，单线程）
            debug: 打印调试信息
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        print(f"\n{'='*50}")
        print(f"Running Evaluation on {eval_file}")
        print(f"  Workers: {workers}")
        print(f"{'='*50}")
        
        # 加载评测数据
        samples = self.data_loader.load_eval_candidates(eval_file)
        
        if max_samples and len(samples) > max_samples:
            print(f"  Limiting to {max_samples} samples")
            random.shuffle(samples)
            samples = samples[:max_samples]
        
        # 调试：检查数据重叠
        if debug:
            self._check_data_overlap(samples)
        
        results: List[EvalResult] = []
        
        if workers <= 1:
            # 单线程模式
            for sample in tqdm(samples, desc="Evaluating"):
                result = self._evaluate_single(sample, top_k, debug=debug)
                results.append(result)
        else:
            # 多线程并发模式（debug 模式下只打印前几个）
            debug_count = 5 if debug else 0
            with ThreadPoolExecutor(max_workers=workers) as executor:
                # 提交所有任务
                future_to_sample = {
                    executor.submit(self._evaluate_single, sample, top_k, debug=(debug and i < debug_count)): sample
                    for i, sample in enumerate(samples)
                }
                
                # 收集结果
                for future in tqdm(as_completed(future_to_sample), total=len(samples), desc=f"Evaluating (workers={workers})"):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        sample = future_to_sample[future]
                        print(f"  Error evaluating {sample.question_id}: {e}")
        
        # 6. 计算汇总指标
        n = len(results)
        
        hits_at_k = {1: 0, 3: 0, 5: 0, 10: 0}
        mrr_sum = 0.0
        correct_count = 0
        total_citation_precision = 0.0
        total_citation_recall = 0.0
        total_hallucination = 0.0
        
        for r in results:
            # 检索指标
            for k in hits_at_k.keys():
                if 0 < r.hit_rank <= k:
                    hits_at_k[k] += 1
            if r.hit_rank > 0:
                mrr_sum += 1.0 / r.hit_rank
            
            # 生成指标
            if r.is_correct:
                correct_count += 1
            total_citation_precision += r.citation_precision
            total_citation_recall += r.citation_recall
            total_hallucination += r.hallucination_rate
        
        avg_precision = total_citation_precision / n if n > 0 else 0
        avg_recall = total_citation_recall / n if n > 0 else 0
        citation_f1 = (2 * avg_precision * avg_recall / (avg_precision + avg_recall)) if (avg_precision + avg_recall) > 0 else 0
        
        metrics = EvalMetrics(
            total_samples=n,
            hit_at_1=hits_at_k[1] / n if n > 0 else 0,
            hit_at_3=hits_at_k[3] / n if n > 0 else 0,
            hit_at_5=hits_at_k[5] / n if n > 0 else 0,
            hit_at_10=hits_at_k[10] / n if n > 0 else 0,
            mrr=mrr_sum / n if n > 0 else 0,
            accuracy=correct_count / n if n > 0 else 0,
            citation_precision=avg_precision,
            citation_recall=avg_recall,
            citation_f1=citation_f1,
            hallucination_rate=total_hallucination / n if n > 0 else 0
        )
        
        return metrics, results
    
    def evaluate_retrieval_only(
        self,
        eval_file: str = "dev_candidates.txt",
        max_samples: int = None,
        top_k: int = 10,
        debug: bool = False
    ) -> EvalMetrics:
        """仅评测检索（不调用LLM生成，速度更快）"""
        print(f"\n{'='*50}")
        print(f"Running Retrieval-Only Evaluation on {eval_file}")
        print(f"{'='*50}")
        
        samples = self.data_loader.load_eval_candidates(eval_file)
        
        if max_samples and len(samples) > max_samples:
            random.shuffle(samples)
            samples = samples[:max_samples]
        
        # 调试：检查数据重叠
        if debug:
            self._check_data_overlap(samples)
        
        hits_at_k = {1: 0, 3: 0, 5: 0, 10: 0}
        mrr_sum = 0.0
        debug_count = 0
        
        for sample in tqdm(samples, desc="Evaluating Retrieval"):
            results = self.rag.retrieve(sample.question_content, top_k=top_k)
            
            hit_rank = -1
            
            # 方式1: 按 ans_id 匹配
            for i, r in enumerate(results):
                ans_id = r.chunk.metadata.get("ans_id", "")
                if ans_id and ans_id in sample.correct_ans_ids:
                    hit_rank = i + 1
                    break
            
            # 方式2: 如果 ans_id 没命中，尝试按 question_id 匹配
            if hit_rank < 0:
                for i, r in enumerate(results):
                    qid = r.chunk.metadata.get("question_id", "")
                    if qid and qid == sample.question_id:
                        hit_rank = i + 1
                        break
            
            # 调试输出
            if debug and hit_rank < 0 and debug_count < 5:
                retrieved_ans_ids = [r.chunk.metadata.get("ans_id", "") for r in results[:5]]
                retrieved_qids = [r.chunk.metadata.get("question_id", "") for r in results[:5]]
                print(f"\n[DEBUG] No hit for question {sample.question_id}:")
                print(f"  Correct ans_ids: {sample.correct_ans_ids}")
                print(f"  Retrieved ans_ids: {retrieved_ans_ids}")
                print(f"  Retrieved question_ids: {retrieved_qids}")
                debug_count += 1
            
            for k in hits_at_k.keys():
                if 0 < hit_rank <= k:
                    hits_at_k[k] += 1
            
            if hit_rank > 0:
                mrr_sum += 1.0 / hit_rank
        
        n = len(samples)
        return EvalMetrics(
            total_samples=n,
            hit_at_1=hits_at_k[1] / n if n > 0 else 0,
            hit_at_3=hits_at_k[3] / n if n > 0 else 0,
            hit_at_5=hits_at_k[5] / n if n > 0 else 0,
            hit_at_10=hits_at_k[10] / n if n > 0 else 0,
            mrr=mrr_sum / n if n > 0 else 0
        )


# ==================== 工具函数 ====================

def print_metrics(metrics: EvalMetrics):
    """打印评测指标"""
    print(f"\n{'='*50}")
    print("Evaluation Results")
    print(f"{'='*50}")
    
    print(f"\n📊 基本信息:")
    print(f"  样本数: {metrics.total_samples}")
    if metrics.elapsed_seconds > 0:
        print(f"  耗时: {metrics.elapsed_seconds:.2f}s")
    
    print(f"\n🔍 检索指标:")
    print(f"  Hit@1:  {metrics.hit_at_1:.4f} ({metrics.hit_at_1*100:.2f}%)")
    print(f"  Hit@3:  {metrics.hit_at_3:.4f} ({metrics.hit_at_3*100:.2f}%)")
    print(f"  Hit@5:  {metrics.hit_at_5:.4f} ({metrics.hit_at_5*100:.2f}%)")
    print(f"  Hit@10: {metrics.hit_at_10:.4f} ({metrics.hit_at_10*100:.2f}%)")
    print(f"  MRR:    {metrics.mrr:.4f}")
    
    if metrics.accuracy > 0 or metrics.citation_f1 > 0:
        print(f"\n📝 生成指标:")
        print(f"  准确率 (Accuracy):     {metrics.accuracy:.4f} ({metrics.accuracy*100:.2f}%)")
        print(f"  引用精确率 (Precision): {metrics.citation_precision:.4f}")
        print(f"  引用召回率 (Recall):    {metrics.citation_recall:.4f}")
        print(f"  引用 F1:               {metrics.citation_f1:.4f}")
        print(f"  幻觉率 (Hallucination): {metrics.hallucination_rate:.4f} ({metrics.hallucination_rate*100:.2f}%)")


def save_results(
    metrics: EvalMetrics,
    results: List[EvalResult],
    output_path: str
):
    """保存评测结果"""
    output = {
        "metrics": asdict(metrics),
        "samples": [
            {
                "question_id": r.question_id,
                "question": r.question,
                "ground_truths": r.ground_truths,
                "prediction": r.prediction,
                "hit": r.hit,
                "hit_rank": r.hit_rank,
                "is_correct": r.is_correct,
                "citation_precision": r.citation_precision,
                "citation_recall": r.citation_recall,
                "hallucination_rate": r.hallucination_rate,
                "retrieved_ans_ids": r.retrieved_ans_ids[:5],
                "llm_judgment": r.llm_judgment
            }
            for r in results
        ]
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")


# ==================== 主函数 ====================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Evaluation Script")
    
    # 数据路径
    parser.add_argument("--data-dir", default="./data", help="数据目录路径")
    
    # LLM 配置
    parser.add_argument("--base-url", default=None, help="LLM API 地址")
    parser.add_argument("--api-key", default=None, help="LLM API 密钥")
    parser.add_argument("--provider", default=None, 
                        choices=["openai", "deepseek", "zhipu", "moonshot", "qwen", "ollama", "vllm"],
                        help="LLM 提供商")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="模型名称")
    
    # 兼容旧参数
    parser.add_argument("--vllm-url", default=None, help="[兼容] 等同于 --base-url")
    
    # 其他 RAG 配置
    parser.add_argument("--embedding", default="BAAI/bge-base-zh-v1.5", help="嵌入模型")
    parser.add_argument("--db-dir", default="./eval_faiss_db", help="FAISS 索引目录")
    
    # 评测配置
    parser.add_argument("--max-knowledge", type=int, default=None, help="最大知识库文档数")
    parser.add_argument("--max-samples", type=int, default=None, help="最大评测样本数")
    parser.add_argument("--top-k", type=int, default=5, help="检索 Top-K")
    parser.add_argument("--batch-size", type=int, default=64, help="批量 embedding 大小")
    parser.add_argument("--workers", type=int, default=1, help="并发线程数（默认 1）")
    parser.add_argument("--eval-file", default="dev_candidates.txt", help="评测文件")
    parser.add_argument("--retrieval-only", action="store_true", help="仅评测检索")
    parser.add_argument("--no-llm-judge", action="store_true", help="不使用LLM评判")
    parser.add_argument("--skip-build", action="store_true", help="跳过知识库构建，使用已有索引")
    parser.add_argument("--debug", action="store_true", help="打印调试信息")
    parser.add_argument("--output", default="eval_results.json", help="结果输出文件")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    # 初始化 RAG
    print("Initializing RAG...")
    print(f"  Provider: {args.provider or 'custom'}")
    print(f"  Base URL: {args.base_url or args.vllm_url or '(from provider)'}")
    print(f"  API Key: {'***' + args.api_key[-4:] if args.api_key else '(not set / from env)'}")
    print(f"  Model: {args.model}")
    
    rag = RAG(
        base_url=args.base_url or args.vllm_url,
        model_name=args.model,
        api_key=args.api_key,
        provider=args.provider,
        embedding_model=args.embedding,
        persist_dir=args.db_dir,
        top_k=args.top_k
    )
    
    # 初始化评测器
    evaluator = RAGEvaluator(
        rag=rag,
        data_dir=args.data_dir,
        use_llm_judge=not args.no_llm_judge,
        seed=args.seed
    )
    
    # 构建知识库（或跳过）
    if args.skip_build:
        # 跳过构建，使用已有索引
        print(f"\n{'='*50}")
        print("Skipping Knowledge Base Build (using existing index)")
        print(f"{'='*50}")
        print(f"  Index path: {args.db_dir}")
        print(f"  Total chunks: {rag.vector_store.count()}")
        
        if rag.vector_store.count() == 0:
            print("  ⚠️ WARNING: Index is empty! Did you forget to build it?")
        
        # 仍需加载问题和答案数据（用于评测）
        evaluator.data_loader.load_questions()
        evaluator.data_loader.load_answers()
    else:
        evaluator.build_knowledge_base(max_docs=args.max_knowledge, batch_size=args.batch_size)
    
    # 执行评测
    start_time = time.time()
    
    if args.retrieval_only:
        metrics = evaluator.evaluate_retrieval_only(
            eval_file=args.eval_file,
            max_samples=args.max_samples,
            top_k=args.top_k,
            debug=args.debug
        )
        results = []
    else:
        metrics, results = evaluator.evaluate(
            eval_file=args.eval_file,
            max_samples=args.max_samples,
            top_k=args.top_k,
            workers=args.workers,
            debug=args.debug
        )
    
    metrics.elapsed_seconds = round(time.time() - start_time, 2)
    
    # 打印结果
    print_metrics(metrics)
    
    # 保存结果
    save_results(metrics, results, args.output)


if __name__ == "__main__":
    main()
