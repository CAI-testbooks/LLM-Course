"""
RAG 评测脚本
使用 HuaTuo 医学百科问答数据集
"""

import json
import random
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import numpy as np

from rag import RAG, Chunk


@dataclass
class EvalSample:
    """评测样本"""
    question: str
    answer: str
    doc_id: int


@dataclass
class EvalResult:
    """评测结果"""
    question: str
    ground_truth: str
    prediction: str
    sources: List[dict]
    retrieval_scores: List[float]
    hit: bool  # 是否检索到相关文档
    

class RAGEvaluator:
    """RAG 评测器"""
    
    def __init__(
        self,
        rag: RAG,
        dataset_name: str = "FreedomIntelligence/huatuo_encyclopedia_qa",
        max_knowledge_docs: int = 5000,  # 知识库文档数量
        eval_samples: int = 100,          # 评测样本数
        seed: int = 42
    ):
        self.rag = rag
        self.dataset_name = dataset_name
        self.max_knowledge_docs = max_knowledge_docs
        self.eval_samples = eval_samples
        self.seed = seed
        
        random.seed(seed)
        np.random.seed(seed)
        
        self.knowledge_ids = set()  # 用于知识库的文档ID
        self.eval_data: List[EvalSample] = []  # 评测数据
    
    def load_dataset(self):
        """加载数据集"""
        print(f"Loading dataset: {self.dataset_name}")
        from datasets import load_dataset
        
        ds = load_dataset(self.dataset_name, split="train")
        print(f"Total samples: {len(ds)}")
        
        return ds
    
    def build_knowledge_base(self, ds) -> int:
        """构建知识库"""
        print(f"\n{'='*50}")
        print("Building Knowledge Base")
        print(f"{'='*50}")
        
        # 打印数据集字段
        print(f"Dataset columns: {ds.column_names}")
        print(f"Sample: {ds[0]}")
        
        # 随机选择文档构建知识库
        all_indices = list(range(len(ds)))
        random.shuffle(all_indices)
        
        knowledge_indices = all_indices[:self.max_knowledge_docs]
        self.knowledge_ids = set(knowledge_indices)
        
        total_chunks = 0
        
        print(f"Adding {len(knowledge_indices)} documents to knowledge base...")
        
        skipped_empty = 0
        skipped_no_chunks = 0
        
        for idx in tqdm(knowledge_indices, desc="Indexing"):
            sample = ds[idx]
            
            # 提取问答对
            question, answer = self._extract_qa(sample)
            
            # 第一条打印调试
            if idx == knowledge_indices[0]:
                print(f"First sample - Q: {question[:50] if question else 'EMPTY'}...")
                print(f"First sample - A: {answer[:50] if answer else 'EMPTY'}...")
            
            if not question or not answer:
                skipped_empty += 1
                continue
            
            # 构建文档内容
            doc_content = f"问题：{question}\n答案：{answer}"
            doc_name = f"doc_{idx}"
            
            # 添加到知识库
            chunks = self.rag.chunker.chunk_text(doc_content, doc_name)
            if not chunks:
                skipped_no_chunks += 1
                continue
            
            # 存储原始问答信息到metadata
            for chunk in chunks:
                chunk.metadata = {
                    "doc_id": idx,
                    "original_question": question,
                    "original_answer": answer
                }
            
            embeddings = self.rag.embedder.embed([c.content for c in chunks])
            self.rag.vector_store.add(chunks, embeddings)
            total_chunks += len(chunks)
        
        print(f"Knowledge base built: {total_chunks} chunks from {len(knowledge_indices)} docs")
        print(f"  Skipped (empty q/a): {skipped_empty}")
        print(f"  Skipped (no chunks): {skipped_no_chunks}")
        return total_chunks
    
    def _extract_qa(self, sample) -> tuple:
        """从样本中提取问答对"""
        question = ""
        answer = ""
        
        # 处理 questions 字段 (嵌套列表 [[q]])
        if "questions" in sample and sample["questions"]:
            q = sample["questions"]
            if isinstance(q, list) and len(q) > 0:
                if isinstance(q[0], list) and len(q[0]) > 0:
                    question = q[0][0]
                else:
                    question = q[0]
        elif "question" in sample:
            question = sample["question"]
        elif "input" in sample:
            question = sample["input"]
        
        # 处理 answers 字段 (列表 [a])
        if "answers" in sample and sample["answers"]:
            a = sample["answers"]
            if isinstance(a, list) and len(a) > 0:
                answer = a[0]
        elif "answer" in sample:
            answer = sample["answer"]
        elif "output" in sample:
            answer = sample["output"]
        
        return question, answer
    
    def prepare_eval_data(self, ds):
        """准备评测数据"""
        print(f"\n{'='*50}")
        print("Preparing Evaluation Data")
        print(f"{'='*50}")
        
        # 从知识库中选择评测样本（确保答案在知识库中）
        knowledge_list = list(self.knowledge_ids)
        random.shuffle(knowledge_list)
        
        eval_indices = knowledge_list[:self.eval_samples]
        
        for idx in eval_indices:
            sample = ds[idx]
            question, answer = self._extract_qa(sample)
            
            if question and answer:
                self.eval_data.append(EvalSample(
                    question=question,
                    answer=answer,
                    doc_id=idx
                ))
        
        print(f"Prepared {len(self.eval_data)} evaluation samples")
    
    def evaluate(self, top_k: int = 5) -> Dict:
        """执行评测"""
        print(f"\n{'='*50}")
        print("Running Evaluation")
        print(f"{'='*50}")
        
        results: List[EvalResult] = []
        
        # 检索评测指标
        retrieval_hits = 0  # 检索命中数
        retrieval_mrr = 0.0  # MRR
        retrieval_recall = 0.0  # Recall@K
        
        for sample in tqdm(self.eval_data, desc="Evaluating"):
            # 执行RAG查询
            try:
                answer, sources = self.rag.query(sample.question, top_k=top_k)
            except Exception as e:
                print(f"Query error: {e}")
                answer = ""
                sources = []
            
            # 检查检索结果
            retrieval_scores = [s["score"] for s in sources]
            
            # 检查是否命中正确文档
            hit = False
            hit_rank = -1
            
            for i, src in enumerate(sources):
                # 检查来源文档名是否匹配
                doc_name = src.get("doc", "")
                if doc_name == f"doc_{sample.doc_id}":
                    hit = True
                    hit_rank = i + 1
                    break
            
            if hit:
                retrieval_hits += 1
                retrieval_mrr += 1.0 / hit_rank
                retrieval_recall += 1.0
            
            results.append(EvalResult(
                question=sample.question,
                ground_truth=sample.answer,
                prediction=answer,
                sources=sources,
                retrieval_scores=retrieval_scores,
                hit=hit
            ))
        
        # 计算指标
        n = len(self.eval_data)
        metrics = {
            "total_samples": n,
            "top_k": top_k,
            "retrieval": {
                "hits": retrieval_hits,
                "hit_rate": retrieval_hits / n if n > 0 else 0,
                "mrr": retrieval_mrr / n if n > 0 else 0,
                "recall@k": retrieval_recall / n if n > 0 else 0,
            }
        }
        
        return metrics, results
    
    def evaluate_retrieval_only(self, top_k: int = 5) -> Dict:
        """仅评测检索（不调用LLM，速度更快）"""
        print(f"\n{'='*50}")
        print("Running Retrieval-Only Evaluation")
        print(f"{'='*50}")
        
        hits_at_k = {1: 0, 3: 0, 5: 0, 10: 0}
        mrr_sum = 0.0
        
        for sample in tqdm(self.eval_data, desc="Evaluating Retrieval"):
            # 仅检索
            results = self.rag.retrieve(sample.question, top_k=max(top_k, 10))
            
            # 查找命中位置
            hit_rank = -1
            for i, r in enumerate(results):
                doc_name = r.chunk.doc_name
                if doc_name == f"doc_{sample.doc_id}":
                    hit_rank = i + 1
                    break
            
            # 统计各K值的命中率
            for k in hits_at_k.keys():
                if 0 < hit_rank <= k:
                    hits_at_k[k] += 1
            
            # MRR
            if hit_rank > 0:
                mrr_sum += 1.0 / hit_rank
        
        n = len(self.eval_data)
        metrics = {
            "total_samples": n,
            "hit@1": hits_at_k[1] / n if n > 0 else 0,
            "hit@3": hits_at_k[3] / n if n > 0 else 0,
            "hit@5": hits_at_k[5] / n if n > 0 else 0,
            "hit@10": hits_at_k[10] / n if n > 0 else 0,
            "mrr": mrr_sum / n if n > 0 else 0,
        }
        
        return metrics
    
    def run(self, retrieval_only: bool = False, top_k: int = 5):
        """运行完整评测流程"""
        # 1. 加载数据集
        ds = self.load_dataset()
        
        # 2. 构建知识库
        self.build_knowledge_base(ds)
        
        # 3. 准备评测数据
        self.prepare_eval_data(ds)
        
        # 4. 执行评测
        if retrieval_only:
            metrics = self.evaluate_retrieval_only(top_k)
            return metrics, None
        else:
            metrics, results = self.evaluate(top_k)
            return metrics, results


def print_metrics(metrics: Dict):
    """打印评测指标"""
    print(f"\n{'='*50}")
    print("Evaluation Results")
    print(f"{'='*50}")
    
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
        else:
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")


def save_results(metrics: Dict, results: List[EvalResult], output_path: str):
    """保存评测结果"""
    output = {
        "metrics": metrics,
        "samples": [
            {
                "question": r.question,
                "ground_truth": r.ground_truth,
                "prediction": r.prediction,
                "hit": r.hit,
                "retrieval_scores": r.retrieval_scores
            }
            for r in (results or [])
        ]
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {output_path}")


# ==================== 主函数 ====================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Evaluation on HuaTuo Dataset")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--embedding", default="BAAI/bge-base-zh-v1.5")
    parser.add_argument("--db-dir", default="./eval_faiss_db")
    parser.add_argument("--knowledge-size", type=int, default=5000, help="知识库文档数量")
    parser.add_argument("--eval-size", type=int, default=100, help="评测样本数量")
    parser.add_argument("--top-k", type=int, default=5, help="检索Top-K")
    parser.add_argument("--retrieval-only", action="store_true", help="仅评测检索")
    parser.add_argument("--output", default="eval_results.json", help="结果输出文件")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # 初始化RAG
    print("Initializing RAG...")
    rag = RAG(
        vllm_url=args.vllm_url,
        model_name=args.model,
        embedding_model=args.embedding,
        persist_dir=args.db_dir,
        top_k=args.top_k
    )
    
    # 清空已有数据，确保评测干净
    rag.vector_store.clear()
    
    # 初始化评测器
    evaluator = RAGEvaluator(
        rag=rag,
        max_knowledge_docs=args.knowledge_size,
        eval_samples=args.eval_size,
        seed=args.seed
    )
    
    # 运行评测
    start_time = time.time()
    metrics, results = evaluator.run(
        retrieval_only=args.retrieval_only,
        top_k=args.top_k
    )
    elapsed = time.time() - start_time
    
    metrics["elapsed_seconds"] = round(elapsed, 2)
    
    # 打印结果
    print_metrics(metrics)
    
    # 保存结果
    save_results(metrics, results, args.output)


if __name__ == "__main__":
    main()