# src/retriever.py
import numpy as np
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer
from typing import List, Dict, Optional
from .config import RAGConfig


class HybridRetriever:
    """混合检索器"""

    def __init__(self, vector_store: FAISS, chunks: List[Dict], config: RAGConfig):
        self.vector_store = vector_store
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.embedding_model)

        # 初始化BM25
        self.chunks = chunks
        self.chunk_texts = [chunk['content'] for chunk in chunks]
        self.bm25 = BM25Okapi([self.tokenize(text)
                              for text in self.chunk_texts])

        # 初始化重排序模型
        self.rerank_model = None
        if config.retrieval_strategy == RetrievalStrategy.RERANK:
            self.init_rerank_model()

    def tokenize(self, text: str) -> List[str]:
        """文本分词"""
        return self.tokenizer.tokenize(text)

    def init_rerank_model(self):
        """初始化重排序模型"""
        from sentence_transformers import CrossEncoder
        self.rerank_model = CrossEncoder('BAAI/bge-reranker-large')

    def dense_retrieve(self, query: str, top_k: int) -> List[Dict]:
        """稠密检索"""
        docs = self.vector_store.similarity_search_with_score(query, k=top_k)
        results = []
        for doc, score in docs:
            results.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': float(score),
                'retrieval_type': 'dense'
            })
        return results

    def sparse_retrieve(self, query: str, top_k: int) -> List[Dict]:
        """稀疏检索（BM25）"""
        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # 获取top_k结果
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                'content': self.chunk_texts[idx],
                'metadata': self.chunks[idx]['metadata'],
                'score': float(scores[idx]),
                'retrieval_type': 'sparse'
            })
        return results

    def hybrid_retrieve(self, query: str, top_k: int, alpha: float = 0.5) -> List[Dict]:
        """混合检索"""
        dense_results = self.dense_retrieve(query, top_k * 2)
        sparse_results = self.sparse_retrieve(query, top_k * 2)

        # 合并结果并去重
        all_results = {}
        for result in dense_results + sparse_results:
            content = result['content']
            if content not in all_results:
                all_results[content] = {
                    'content': content,
                    'metadata': result['metadata'],
                    'dense_score': 0.0,
                    'sparse_score': 0.0,
                    'combined_score': 0.0
                }

            if result['retrieval_type'] == 'dense':
                all_results[content]['dense_score'] = result['score']
            else:
                all_results[content]['sparse_score'] = result['score']

        # 归一化分数并计算综合分数
        max_dense = max(r['dense_score'] for r in all_results.values()) or 1
        max_sparse = max(r['sparse_score'] for r in all_results.values()) or 1

        for content in all_results:
            all_results[content]['dense_score_norm'] = all_results[content]['dense_score'] / max_dense
            all_results[content]['sparse_score_norm'] = all_results[content]['sparse_score'] / max_sparse
            all_results[content]['combined_score'] = (
                alpha * all_results[content]['dense_score_norm'] +
                (1 - alpha) * all_results[content]['sparse_score_norm']
            )

        # 按综合分数排序
        sorted_results = sorted(all_results.values(),
                                key=lambda x: x['combined_score'],
                                reverse=True)[:top_k]

        return [{
            'content': r['content'],
            'metadata': r['metadata'],
            'score': r['combined_score'],
            'retrieval_type': 'hybrid'
        } for r in sorted_results]

    def retrieve_with_rerank(self, query: str, top_k: int) -> List[Dict]:
        """带重排序的检索"""
        # 第一阶段：混合检索获取更多候选
        candidate_results = self.hybrid_retrieve(query, top_k * 3)

        # 第二阶段：重排序
        if self.rerank_model:
            pairs = [(query, r['content']) for r in candidate_results]
            rerank_scores = self.rerank_model.predict(pairs)

            for result, score in zip(candidate_results, rerank_scores):
                result['rerank_score'] = float(score)

            # 按重排序分数排序
            candidate_results.sort(
                key=lambda x: x['rerank_score'], reverse=True)

        return candidate_results[:top_k]

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """主检索方法"""
        if top_k is None:
            top_k = self.config.top_k

        strategy = self.config.retrieval_strategy

        if strategy == RetrievalStrategy.DENSE:
            return self.dense_retrieve(query, top_k)
        elif strategy == RetrievalStrategy.SPARSE:
            return self.sparse_retrieve(query, top_k)
        elif strategy == RetrievalStrategy.HYBRID:
            return self.hybrid_retrieve(query, top_k)
        elif strategy == RetrievalStrategy.RERANK:
            return self.retrieve_with_rerank(query, top_k)
        else:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")
