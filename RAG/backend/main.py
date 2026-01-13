"""
RAG 医学问答系统 - FastAPI 后端
支持多轮对话、流式输出、来源引用
"""

import os
import sys
import json
import uuid
from pathlib import Path
from typing import List, Dict, Optional, AsyncGenerator
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# 添加项目根目录到路径，以便导入 rag 模块
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))  # 添加当前目录

from rag import RAG
from rag.prompts import (
    get_prompt,
    build_rag_prompt,
    should_refuse,
    analyze_query as do_analyze_query
)


# ==================== 配置 ====================

class Config:
    """应用配置"""
    # LLM 配置
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", os.getenv("VLLM_URL", ""))
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "")  # openai/deepseek/zhipu/moonshot/qwen/ollama/vllm
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    
    # Embedding 配置
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-zh-v1.5")
    
    # 存储配置
    DB_DIR: str = os.getenv("DB_DIR", str(PROJECT_ROOT / "data" / "faiss_db"))
    
    # 检索配置
    TOP_K: int = int(os.getenv("TOP_K", "5"))
    RELEVANCE_THRESHOLD: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.45"))
    MAX_HISTORY: int = int(os.getenv("MAX_HISTORY", "10"))
    
    # 查询重写配置
    ENABLE_REWRITE: bool = os.getenv("ENABLE_REWRITE", "false").lower() == "true"
    REWRITE_MODE: str = os.getenv("REWRITE_MODE", "single")  # single/multi/context/auto/hyde
    ENABLE_CACHE: bool = os.getenv("ENABLE_CACHE", "true").lower() == "true"
    CACHE_SIZE: int = int(os.getenv("CACHE_SIZE", "1000"))
    
    # 重排序配置
    ENABLE_RERANK: bool = os.getenv("ENABLE_RERANK", "false").lower() == "true"
    RERANK_MODEL: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
    RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "0")) or None  # 0 表示使用 TOP_K


config = Config()


# ==================== Pydantic 模型 ====================

class Message(BaseModel):
    """消息"""
    role: str = Field(..., description="角色: user 或 assistant")
    content: str = Field(..., description="消息内容")


class ChatRequest(BaseModel):
    """聊天请求"""
    message: str = Field(..., description="用户消息")
    conversation_id: Optional[str] = Field(None, description="会话ID")
    top_k: Optional[int] = Field(None, description="检索数量")
    # 查询重写参数
    enable_rewrite: Optional[bool] = Field(None, description="是否启用查询重写")
    rewrite_mode: Optional[str] = Field(None, description="重写模式: single/multi/context/auto/hyde")
    # 重排序参数
    enable_rerank: Optional[bool] = Field(None, description="是否启用重排序")


class Source(BaseModel):
    """引用来源"""
    index: int
    doc_name: str
    score: float
    content: str
    metadata: Optional[dict] = None


class ChatResponse(BaseModel):
    """聊天响应"""
    conversation_id: str
    message: str
    sources: List[Source]
    refused: bool = False
    created_at: str


class ConversationInfo(BaseModel):
    """会话信息"""
    conversation_id: str
    message_count: int
    created_at: str
    last_message_at: str
    preview: str


class StatsResponse(BaseModel):
    """统计信息响应"""
    total_chunks: int
    embedding_model: str
    embedding_dim: int
    active_conversations: int


# ==================== 会话管理 ====================

class ConversationManager:
    """会话管理器 - 内存存储"""
    
    def __init__(self, max_history: int = 10):
        self.conversations: Dict[str, List[Message]] = {}
        self.metadata: Dict[str, dict] = {}
        self.max_history = max_history
    
    def create(self) -> str:
        """创建新会话"""
        conv_id = str(uuid.uuid4())[:8]
        self.conversations[conv_id] = []
        self.metadata[conv_id] = {
            "created_at": datetime.now().isoformat(),
            "last_message_at": datetime.now().isoformat()
        }
        return conv_id
    
    def get(self, conv_id: str) -> List[Message]:
        """获取会话历史"""
        return self.conversations.get(conv_id, [])
    
    def add_message(self, conv_id: str, role: str, content: str):
        """添加消息"""
        if conv_id not in self.conversations:
            self.conversations[conv_id] = []
            self.metadata[conv_id] = {
                "created_at": datetime.now().isoformat(),
                "last_message_at": datetime.now().isoformat()
            }
        
        self.conversations[conv_id].append(Message(role=role, content=content))
        self.metadata[conv_id]["last_message_at"] = datetime.now().isoformat()
        
        # 限制历史长度
        max_messages = self.max_history * 2
        if len(self.conversations[conv_id]) > max_messages:
            self.conversations[conv_id] = self.conversations[conv_id][-max_messages:]
    
    def get_history_for_prompt(self, conv_id: str) -> List[dict]:
        """获取用于提示词的历史"""
        messages = self.get(conv_id)
        return [{"role": m.role, "content": m.content} for m in messages]
    
    def delete(self, conv_id: str):
        """删除会话"""
        self.conversations.pop(conv_id, None)
        self.metadata.pop(conv_id, None)
    
    def list_all(self) -> List[ConversationInfo]:
        """列出所有会话"""
        result = []
        for conv_id, messages in self.conversations.items():
            meta = self.metadata.get(conv_id, {})
            preview = messages[-1].content[:50] + "..." if messages else ""
            result.append(ConversationInfo(
                conversation_id=conv_id,
                message_count=len(messages),
                created_at=meta.get("created_at", ""),
                last_message_at=meta.get("last_message_at", ""),
                preview=preview
            ))
        return sorted(result, key=lambda x: x.last_message_at, reverse=True)
    
    def clear_all(self):
        """清空所有会话"""
        self.conversations.clear()
        self.metadata.clear()


# ==================== RAG 服务 ====================

class RAGService:
    """RAG 服务封装"""
    
    def __init__(self):
        self._rag: Optional[RAG] = None
    
    @property
    def rag(self) -> RAG:
        """延迟初始化 RAG"""
        if self._rag is None:
            print("=" * 50)
            print("Initializing RAG Service")
            print("=" * 50)
            print(f"  LLM Provider: {config.LLM_PROVIDER or 'custom'}")
            print(f"  LLM Base URL: {config.LLM_BASE_URL or '(from provider)'}")
            print(f"  LLM API Key: {'***' + config.LLM_API_KEY[-4:] if config.LLM_API_KEY else '(not set)'}")
            print(f"  Model: {config.MODEL_NAME}")
            print(f"  Embedding: {config.EMBEDDING_MODEL}")
            print(f"  DB Dir: {config.DB_DIR}")
            print(f"  Query Rewrite: {config.ENABLE_REWRITE} ({config.REWRITE_MODE})")
            print(f"  Cache: {config.ENABLE_CACHE} (size: {config.CACHE_SIZE})")
            print(f"  Rerank: {config.ENABLE_RERANK} ({config.RERANK_MODEL})")
            
            # 确保目录存在
            Path(config.DB_DIR).mkdir(parents=True, exist_ok=True)
            
            self._rag = RAG(
                base_url=config.LLM_BASE_URL or None,
                model_name=config.MODEL_NAME,
                api_key=config.LLM_API_KEY or None,
                provider=config.LLM_PROVIDER or None,
                embedding_model=config.EMBEDDING_MODEL,
                persist_dir=config.DB_DIR,
                top_k=config.TOP_K,
                enable_rewrite=config.ENABLE_REWRITE,
                rewrite_mode=config.REWRITE_MODE,
                enable_cache=config.ENABLE_CACHE,
                cache_size=config.CACHE_SIZE,
                enable_rerank=config.ENABLE_RERANK,
                rerank_model=config.RERANK_MODEL,
                rerank_top_k=config.RERANK_TOP_K
            )
            print(f"  Total chunks: {self._rag.stats()['total_chunks']}")
            print("=" * 50)
        return self._rag
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = None,
        enable_rewrite: bool = None,
        rewrite_mode: str = None,
        enable_rerank: bool = None,
        chat_history: List[dict] = None
    ) -> List[dict]:
        """检索（支持重写和重排序）"""
        top_k = top_k or config.TOP_K
        results = self.rag.retrieve(
            query, 
            top_k=top_k,
            enable_rewrite=enable_rewrite,
            rewrite_mode=rewrite_mode,
            enable_rerank=enable_rerank,
            chat_history=chat_history
        )
        
        return [
            {
                "index": i + 1,
                "doc_name": r.chunk.doc_name,
                "score": round(r.score, 4),
                "content": r.chunk.content,
                "metadata": r.chunk.metadata
            }
            for i, r in enumerate(results)
        ]
    
    def retrieve_with_details(
        self,
        query: str,
        top_k: int = None,
        enable_rewrite: bool = None,
        rewrite_mode: str = None,
        enable_rerank: bool = None,
        chat_history: List[dict] = None
    ) -> dict:
        """检索并返回详细信息"""
        top_k = top_k or config.TOP_K
        details = self.rag.retrieve_with_details(
            query,
            top_k=top_k,
            enable_rewrite=enable_rewrite,
            rewrite_mode=rewrite_mode,
            enable_rerank=enable_rerank,
            chat_history=chat_history
        )
        
        # 转换结果格式
        results = [
            {
                "index": i + 1,
                "doc_name": r.chunk.doc_name,
                "score": round(r.score, 4),
                "content": r.chunk.content,
                "metadata": r.chunk.metadata
            }
            for i, r in enumerate(details["results"])
        ]
        
        return {
            "original_query": details["original_query"],
            "rewritten_queries": details["rewritten_queries"],
            "rewrite_mode": details["rewrite_mode"],
            "results": results,
            "reranked": details["reranked"]
        }
    
    def generate(
        self,
        question: str,
        retrieval_results: List[dict],
        chat_history: List[dict] = None
    ) -> str:
        """生成回答"""
        prompt = build_rag_prompt(
            question=question,
            retrieval_results=retrieval_results,
            chat_history=chat_history
        )
        system_prompt = get_prompt("system")
        return self.rag.llm.generate(prompt, system_prompt=system_prompt)
    
    def generate_stream(
        self,
        question: str,
        retrieval_results: List[dict],
        chat_history: List[dict] = None
    ):
        """流式生成"""
        prompt = build_rag_prompt(
            question=question,
            retrieval_results=retrieval_results,
            chat_history=chat_history
        )
        system_prompt = get_prompt("system")
        
        for chunk in self.rag.llm.generate_stream(prompt, system_prompt=system_prompt):
            yield chunk
    
    def stats(self) -> dict:
        """获取统计"""
        return self.rag.stats()


# ==================== 全局实例 ====================

rag_service = RAGService()
conversation_manager = ConversationManager(max_history=config.MAX_HISTORY)


# ==================== FastAPI 应用 ====================

app = FastAPI(
    title="RAG 医学问答系统 API",
    description="基于知识库的医学问答后端服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 配置 - 允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== API 路由 ====================

@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """获取系统统计"""
    try:
        stats = rag_service.stats()
        return StatsResponse(
            total_chunks=stats["total_chunks"],
            embedding_model=stats["embedding_model"],
            embedding_dim=stats["embedding_dim"],
            active_conversations=len(conversation_manager.conversations)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """聊天接口（非流式）"""
    try:
        # 获取或创建会话
        conv_id = request.conversation_id or conversation_manager.create()
        history = conversation_manager.get_history_for_prompt(conv_id)
        
        # 检索（支持重写和重排序）
        retrieval_results = rag_service.retrieve(
            request.message,
            top_k=request.top_k or config.TOP_K,
            enable_rewrite=request.enable_rewrite,
            rewrite_mode=request.rewrite_mode,
            enable_rerank=request.enable_rerank,
            chat_history=history
        )
        
        # 判断是否拒绝回答
        refused = should_refuse(retrieval_results, config.RELEVANCE_THRESHOLD)
        
        if refused:
            answer = get_prompt("no_context")
        else:
            answer = rag_service.generate(request.message, retrieval_results, history)
        
        # 保存消息
        conversation_manager.add_message(conv_id, "user", request.message)
        conversation_manager.add_message(conv_id, "assistant", answer)
        
        # 构建来源
        sources = [
            Source(
                index=r["index"],
                doc_name=r["doc_name"],
                score=r["score"],
                content=r["content"][:300] + "..." if len(r["content"]) > 300 else r["content"],
                metadata=r.get("metadata")
            )
            for r in retrieval_results
        ]
        
        return ChatResponse(
            conversation_id=conv_id,
            message=answer,
            sources=sources,
            refused=refused,
            created_at=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """聊天接口（流式）"""
    try:
        conv_id = request.conversation_id or conversation_manager.create()
        history = conversation_manager.get_history_for_prompt(conv_id)
        
        retrieval_results = rag_service.retrieve(
            request.message,
            top_k=request.top_k or config.TOP_K,
            enable_rewrite=request.enable_rewrite,
            rewrite_mode=request.rewrite_mode,
            enable_rerank=request.enable_rerank,
            chat_history=history
        )
        
        refused = should_refuse(retrieval_results, config.RELEVANCE_THRESHOLD)
        
        def generate():
            full_response = ""
            
            # 发送元数据
            sources = [
                {
                    "index": r["index"],
                    "doc_name": r["doc_name"],
                    "score": r["score"],
                    "content": r["content"][:300] + "..." if len(r["content"]) > 300 else r["content"]
                }
                for r in retrieval_results
            ]
            
            meta = {
                "type": "meta",
                "conversation_id": conv_id,
                "sources": sources,
                "refused": refused
            }
            yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"
            
            if refused:
                answer = get_prompt("no_context")
                yield f"data: {json.dumps({'type': 'content', 'content': answer}, ensure_ascii=False)}\n\n"
                full_response = answer
            else:
                for chunk in rag_service.generate_stream(request.message, retrieval_results, history):
                    full_response += chunk
                    yield f"data: {json.dumps({'type': 'content', 'content': chunk}, ensure_ascii=False)}\n\n"
            
            # 保存消息
            conversation_manager.add_message(conv_id, "user", request.message)
            conversation_manager.add_message(conv_id, "assistant", full_response)
            
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/conversations", response_model=List[ConversationInfo])
async def list_conversations():
    """列出所有会话"""
    return conversation_manager.list_all()


@app.get("/api/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    """获取会话详情"""
    messages = conversation_manager.get(conv_id)
    if not messages and conv_id not in conversation_manager.conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        "conversation_id": conv_id,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "metadata": conversation_manager.metadata.get(conv_id, {})
    }


@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    """删除会话"""
    conversation_manager.delete(conv_id)
    return {"status": "ok", "deleted": conv_id}


@app.post("/api/conversations/clear")
async def clear_conversations():
    """清空所有会话"""
    conversation_manager.clear_all()
    return {"status": "ok"}


@app.post("/api/retrieve")
async def retrieve(request: ChatRequest):
    """检索接口（支持重写和重排序）"""
    try:
        results = rag_service.retrieve(
            request.message,
            top_k=request.top_k or config.TOP_K,
            enable_rewrite=request.enable_rewrite,
            rewrite_mode=request.rewrite_mode,
            enable_rerank=request.enable_rerank
        )
        return {"query": request.message, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/retrieve/details")
async def retrieve_with_details(request: ChatRequest):
    """检索接口（返回详细信息，包括重写后的查询）"""
    try:
        # 获取对话历史（如果有）
        chat_history = None
        if request.conversation_id:
            chat_history = conversation_manager.get_history_for_prompt(request.conversation_id)
        
        details = rag_service.retrieve_with_details(
            request.message,
            top_k=request.top_k or config.TOP_K,
            enable_rewrite=request.enable_rewrite,
            rewrite_mode=request.rewrite_mode,
            enable_rerank=request.enable_rerank,
            chat_history=chat_history
        )
        return details
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class RewriteRequest(BaseModel):
    """查询重写请求"""
    query: str = Field(..., description="原始查询")
    mode: str = Field("single", description="重写模式: single/multi/context")
    conversation_id: Optional[str] = Field(None, description="会话ID（context模式需要）")


@app.post("/api/rewrite")
async def rewrite_query(request: RewriteRequest):
    """查询重写接口"""
    try:
        rewriter = rag_service.rag.query_rewriter
        
        if request.mode == "single":
            rewritten = rewriter.rewrite(request.query)
            return {
                "original": request.query,
                "rewritten": [rewritten],
                "mode": "single"
            }
        elif request.mode == "multi":
            queries = rewriter.generate_multi_queries(request.query)
            return {
                "original": request.query,
                "rewritten": queries,
                "mode": "multi"
            }
        elif request.mode == "context":
            history = []
            if request.conversation_id:
                history = conversation_manager.get_history_for_prompt(request.conversation_id)
            rewritten = rewriter.rewrite_with_context(request.query, history)
            return {
                "original": request.query,
                "rewritten": [rewritten],
                "mode": "context"
            }
        else:
            raise HTTPException(status_code=400, detail=f"Unknown mode: {request.mode}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class AnalyzeRequest(BaseModel):
    """查询分析请求"""
    query: str = Field(..., description="查询")


@app.post("/api/analyze")
async def analyze_query(request: AnalyzeRequest):
    """分析查询，推荐重写模式"""
    try:
        analysis = do_analyze_query(request.query)
        return {
            "query": request.query,
            "features": analysis["features"],
            "recommended_mode": analysis["recommended_mode"],
            "reason": analysis["reason"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/cache/stats")
async def get_cache_stats():
    """获取查询缓存统计信息"""
    try:
        stats = rag_service.rag.get_cache_stats()
        if stats is None:
            return {
                "enabled": False,
                "message": "缓存未启用"
            }
        return {
            "enabled": True,
            **stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/cache/clear")
async def clear_cache():
    """清空查询缓存"""
    try:
        rag_service.rag.clear_cache()
        return {"success": True, "message": "缓存已清空"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class HyDERequest(BaseModel):
    """HyDE 请求"""
    query: str = Field(..., description="用户查询")
    short: bool = Field(False, description="是否生成短文档")


@app.post("/api/hyde")
async def generate_hyde_document(request: HyDERequest):
    """生成 HyDE 假设文档"""
    try:
        hyde_doc = rag_service.rag.query_rewriter.generate_hyde_document(
            request.query, 
            short=request.short
        )
        return {
            "query": request.query,
            "hyde_document": hyde_doc,
            "short": request.short
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 启动入口 ====================

def main():
    """启动服务"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG API Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--reload", action="store_true")
    
    # LLM 配置
    parser.add_argument("--base-url", default=None, help="LLM API 地址")
    parser.add_argument("--api-key", default=None, help="LLM API 密钥")
    parser.add_argument("--provider", default=None, 
                        choices=["openai", "deepseek", "zhipu", "moonshot", "qwen", "ollama", "vllm"],
                        help="LLM 提供商")
    parser.add_argument("--model", default=None, help="模型名称")
    
    # 兼容旧参数
    parser.add_argument("--vllm-url", default=None, help="[兼容] 等同于 --base-url")
    
    # 其他配置
    parser.add_argument("--embedding", default=None, help="嵌入模型")
    parser.add_argument("--db-dir", default=None, help="FAISS 索引目录")
    
    # 查询重写配置
    parser.add_argument("--enable-rewrite", action="store_true", help="启用查询重写")
    parser.add_argument("--rewrite-mode", default=None, 
                        choices=["single", "multi", "context", "auto", "hyde", "hyde_short"],
                        help="查询重写模式")
    parser.add_argument("--enable-cache", action="store_true", default=None, help="启用查询缓存")
    parser.add_argument("--no-cache", action="store_true", help="禁用查询缓存")
    parser.add_argument("--cache-size", type=int, default=None, help="缓存大小")
    
    # 重排序配置
    parser.add_argument("--enable-rerank", action="store_true", help="启用重排序")
    parser.add_argument("--rerank-model", default=None, help="重排序模型")
    parser.add_argument("--rerank-top-k", type=int, default=None, help="重排序后返回数量")
    
    args = parser.parse_args()
    
    # 覆盖配置
    if args.base_url or args.vllm_url:
        config.LLM_BASE_URL = args.base_url or args.vllm_url
    if args.api_key:
        config.LLM_API_KEY = args.api_key
    if args.provider:
        config.LLM_PROVIDER = args.provider
    if args.model:
        config.MODEL_NAME = args.model
    if args.embedding:
        config.EMBEDDING_MODEL = args.embedding
    if args.db_dir:
        config.DB_DIR = args.db_dir
    
    # 查询重写配置
    if args.enable_rewrite:
        config.ENABLE_REWRITE = True
    if args.rewrite_mode:
        config.REWRITE_MODE = args.rewrite_mode
    
    # 缓存配置
    if args.enable_cache:
        config.ENABLE_CACHE = True
    if args.no_cache:
        config.ENABLE_CACHE = False
    if args.cache_size:
        config.CACHE_SIZE = args.cache_size
    
    # 重排序配置
    if args.enable_rerank:
        config.ENABLE_RERANK = True
    if args.rerank_model:
        config.RERANK_MODEL = args.rerank_model
    if args.rerank_top_k:
        config.RERANK_TOP_K = args.rerank_top_k
    
    print(f"\n🚀 Starting RAG API Server")
    print(f"   URL: http://{args.host}:{args.port}")
    print(f"   Docs: http://{args.host}:{args.port}/docs")
    if config.ENABLE_REWRITE:
        print(f"   Query Rewrite: {config.REWRITE_MODE}")
    if config.ENABLE_CACHE:
        print(f"   Cache: enabled (size: {config.CACHE_SIZE})")
    if config.ENABLE_RERANK:
        print(f"   Rerank: {config.RERANK_MODEL}")
    print()
    
    uvicorn.run(
        "main:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()