# src/api_app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import time
from .rag_system import RAGSystem


class QueryRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    top_k: Optional[int] = None
    temperature: Optional[float] = None


class QueryResponse(BaseModel):
    answer: str
    references: List[Dict]
    confidence: float
    uncertain: bool
    latency: float


class FastAPIApp:
    """FastAPI后端应用"""

    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system
        self.app = FastAPI(title="RAG API", version="1.0.0")
        self.setup_routes()

    def setup_routes(self):
        """设置路由"""

        @self.app.get("/")
        async def root():
            return {"message": "RAG API Service", "status": "running"}

        @self.app.post("/query", response_model=QueryResponse)
        async def query(request: QueryRequest):
            start_time = time.time()

            # 临时更新配置
            if request.top_k:
                self.rag_system.config.top_k = request.top_k
            if request.temperature:
                self.rag_system.config.temperature = request.temperature

            # 获取回答
            result = self.rag_system.answer(
                request.query, request.conversation_id)

            latency = time.time() - start_time

            return QueryResponse(
                answer=result['answer'],
                references=result['references'],
                confidence=result['confidence'],
                uncertain=result['uncertain'],
                latency=latency
            )

        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp": time.time()}

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """运行API服务"""
        uvicorn.run(self.app, host=host, port=port)
