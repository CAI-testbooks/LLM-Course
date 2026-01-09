"""
RAG 核心模块
"""

from .rag import RAG, Chunk, RetrievalResult, Chunker, Embedder, FAISSStore, LLMClient
from .prompts import (
    PROMPTS,
    get_prompt,
    format_chat_history,
    format_context,
    build_rag_prompt,
    should_refuse
)

__all__ = [
    # RAG 核心类
    "RAG",
    "Chunk",
    "RetrievalResult",
    "Chunker",
    "Embedder",
    "FAISSStore",
    "LLMClient",
    # 提示词相关
    "PROMPTS",
    "get_prompt",
    "format_chat_history",
    "format_context",
    "build_rag_prompt",
    "should_refuse",
]
