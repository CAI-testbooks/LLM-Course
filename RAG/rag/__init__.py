"""
医疗问答 RAG 系统
支持多轮对话、来源引用、查询重写、HyDE、重排序、缓存
"""

from .rag import (
    RAG,
    Chunk,
    RetrievalResult,
    Chunker,
    Embedder,
    FAISSStore,
    LLMClient,
    QueryRewriter,
    CachedQueryRewriter,
    QueryCache,
    Reranker,
)

from .prompts import (
    get_prompt,
    format_chat_history,
    format_context,
    build_rag_prompt,
    build_rewrite_prompt,
    should_refuse,
    analyze_query,
    PROMPTS,
)

__version__ = "2.1.0"
__all__ = [
    # 核心类
    "RAG",
    "Chunk",
    "RetrievalResult",
    "Chunker",
    "Embedder",
    "FAISSStore",
    "LLMClient",
    # 查询处理
    "QueryRewriter",
    "CachedQueryRewriter",
    "QueryCache",
    "Reranker",
    # 提示词相关
    "get_prompt",
    "format_chat_history",
    "format_context",
    "build_rag_prompt",
    "build_rewrite_prompt",
    "should_refuse",
    "analyze_query",
    "PROMPTS",
]
