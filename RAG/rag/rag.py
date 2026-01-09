"""
简单RAG系统 - 基于 FAISS
vLLM + FAISS + BGE Embedding
"""

import os
import json
import pickle
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, asdict
import numpy as np

# ==================== 数据结构 ====================

@dataclass
class Chunk:
    """文档块"""
    content: str
    chunk_id: str
    doc_name: str
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass 
class RetrievalResult:
    """检索结果"""
    chunk: Chunk
    score: float


# ==================== 文档分块 ====================

class Chunker:
    """简单的文档分块器"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, doc_name: str = "unknown") -> List[Chunk]:
        """将文本分块"""
        if not text.strip():
            return []
        
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        chunk_idx = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(current_chunk) + len(para) + 1 <= self.chunk_size:
                current_chunk += ("\n" if current_chunk else "") + para
            else:
                if current_chunk:
                    chunk_id = f"{hashlib.md5(doc_name.encode()).hexdigest()[:8]}_{chunk_idx}"
                    chunks.append(Chunk(
                        content=current_chunk,
                        chunk_id=chunk_id,
                        doc_name=doc_name
                    ))
                    chunk_idx += 1
                
                if len(para) > self.chunk_size:
                    sentences = self._split_sentences(para)
                    current_chunk = ""
                    for sent in sentences:
                        if len(current_chunk) + len(sent) + 1 <= self.chunk_size:
                            current_chunk += (" " if current_chunk else "") + sent
                        else:
                            if current_chunk:
                                chunk_id = f"{hashlib.md5(doc_name.encode()).hexdigest()[:8]}_{chunk_idx}"
                                chunks.append(Chunk(
                                    content=current_chunk,
                                    chunk_id=chunk_id,
                                    doc_name=doc_name
                                ))
                                chunk_idx += 1
                            current_chunk = sent
                else:
                    if chunks and self.chunk_overlap > 0:
                        overlap = chunks[-1].content[-self.chunk_overlap:]
                        current_chunk = overlap + "\n" + para
                    else:
                        current_chunk = para
        
        if current_chunk:
            chunk_id = f"{hashlib.md5(doc_name.encode()).hexdigest()[:8]}_{chunk_idx}"
            chunks.append(Chunk(
                content=current_chunk,
                chunk_id=chunk_id,
                doc_name=doc_name
            ))
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """分句"""
        import re
        sentences = re.split(r'(?<=[。！？.!?])\s*', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk_file(self, file_path: str) -> List[Chunk]:
        """从文件加载并分块"""
        path = Path(file_path)
        
        if path.suffix.lower() == '.pdf':
            try:
                from pypdf import PdfReader
                reader = PdfReader(path)
                text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
            except ImportError:
                print("Warning: pypdf not installed")
                return []
        else:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
        
        return self.chunk_text(text, path.name)


# ==================== 向量嵌入 ====================

class Embedder:
    """向量嵌入模型"""
    
    def __init__(self, model_name: str = "BAAI/bge-base-zh-v1.5"):
        self.model_name = model_name
        self._model = None
    
    @property
    def model(self):
        if self._model is None:
            print(f"Loading embedding model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    @property
    def dim(self) -> int:
        """向量维度"""
        return self.model.get_sentence_embedding_dimension()
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """生成嵌入向量"""
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, normalize_embeddings=True).astype('float32')
    
    def embed_query(self, query: str) -> np.ndarray:
        """生成查询嵌入"""
        if "bge" in self.model_name.lower():
            query = f"为这个句子生成表示以用于检索相关文章：{query}"
        return self.embed([query])[0]


# ==================== FAISS 向量存储 ====================

class FAISSStore:
    """基于FAISS的向量存储"""
    
    def __init__(self, persist_dir: str = "./faiss_db", dim: int = 768):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.dim = dim
        self.index = None
        self.chunks: List[Chunk] = []
        
        self.index_path = self.persist_dir / "index.faiss"
        self.chunks_path = self.persist_dir / "chunks.pkl"
        self._dirty = False  # 标记是否有未保存的更改
        
        self._load()
    
    def _load(self):
        """加载已有索引"""
        import faiss
        
        if self.index_path.exists() and self.chunks_path.exists():
            print(f"Loading FAISS index from {self.persist_dir}")
            self.index = faiss.read_index(str(self.index_path))
            with open(self.chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
            print(f"Loaded {len(self.chunks)} chunks")
        else:
            # IndexFlatIP: 内积（向量归一化后等于余弦相似度）
            self.index = faiss.IndexFlatIP(self.dim)
            self.chunks = []
    
    def save(self):
        """保存索引到磁盘"""
        if not self._dirty and self.index_path.exists():
            return  # 没有更改，跳过
        
        import faiss
        faiss.write_index(self.index, str(self.index_path))
        with open(self.chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        self._dirty = False
        print(f"Saved FAISS index ({len(self.chunks)} chunks)")
    
    def add(self, chunks: List[Chunk], embeddings: np.ndarray, auto_save: bool = False):
        """
        添加文档块
        
        Args:
            chunks: 文档块列表
            embeddings: 嵌入向量
            auto_save: 是否自动保存到磁盘（默认 False，建议批量添加完成后手动调用 save）
        """
        if len(chunks) == 0:
            return
        
        embeddings = embeddings.astype('float32')
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        self._dirty = True
        
        if auto_save:
            self.save()
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """搜索相似文档"""
        if self.index.ntotal == 0:
            return []
        
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def count(self) -> int:
        return len(self.chunks)
    
    def clear(self):
        """清空数据"""
        import faiss
        self.index = faiss.IndexFlatIP(self.dim)
        self.chunks = []
        self._dirty = False
        if self.index_path.exists():
            self.index_path.unlink()
        if self.chunks_path.exists():
            self.chunks_path.unlink()


# ==================== LLM 客户端 ====================

class LLMClient:
    """
    OpenAI 兼容的 LLM 客户端
    支持 OpenAI、vLLM、Ollama、DeepSeek、智谱等 OpenAI 兼容 API
    """
    
    # 常用 API 地址
    PROVIDERS = {
        "openai": "https://api.openai.com/v1",
        "deepseek": "https://api.deepseek.com/v1",
        "zhipu": "https://open.bigmodel.cn/api/paas/v4",
        "moonshot": "https://api.moonshot.cn/v1",
        "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "ollama": "http://localhost:11434/v1",
        "vllm": "http://localhost:8000/v1",
    }
    
    def __init__(
        self, 
        base_url: str = None,
        model: str = "gpt-3.5-turbo",
        api_key: str = None,
        provider: str = None,
        timeout: float = 120.0
    ):
        """
        初始化 LLM 客户端
        
        Args:
            base_url: API 地址，如 https://api.openai.com/v1
            model: 模型名称
            api_key: API 密钥（可选，本地部署不需要）
            provider: 提供商名称（openai/deepseek/zhipu/moonshot/qwen/ollama/vllm）
            timeout: 请求超时时间
        
        Examples:
            # 使用 OpenAI
            client = LLMClient(provider="openai", api_key="sk-xxx", model="gpt-4")
            
            # 使用 DeepSeek
            client = LLMClient(provider="deepseek", api_key="sk-xxx", model="deepseek-chat")
            
            # 使用本地 vLLM
            client = LLMClient(base_url="http://localhost:8000/v1", model="Qwen/Qwen2.5-7B")
            
            # 使用 Ollama
            client = LLMClient(provider="ollama", model="llama3")
        """
        # 确定 base_url
        if base_url:
            self.base_url = base_url.rstrip("/")
        elif provider and provider.lower() in self.PROVIDERS:
            self.base_url = self.PROVIDERS[provider.lower()]
        else:
            self.base_url = self.PROVIDERS["vllm"]  # 默认本地 vLLM
        
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
        self.timeout = timeout
        
        # 构建请求头
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.7
    ) -> str:
        """生成回复"""
        import httpx
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            },
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.7
    ):
        """流式生成"""
        import httpx
        import json as json_lib
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        with httpx.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True
            },
            timeout=self.timeout
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json_lib.loads(data)
                        content = chunk["choices"][0].get("delta", {}).get("content", "")
                        if content:
                            yield content
                    except json_lib.JSONDecodeError:
                        continue
    
    def chat(
        self,
        messages: List[dict],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        stream: bool = False
    ):
        """
        多轮对话
        
        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            max_tokens: 最大生成长度
            temperature: 温度
            stream: 是否流式输出
        """
        import httpx
        import json as json_lib
        
        if stream:
            def stream_generator():
                with httpx.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json={
                        "model": self.model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "stream": True
                    },
                    timeout=self.timeout
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data.strip() == "[DONE]":
                                break
                            try:
                                chunk = json_lib.loads(data)
                                content = chunk["choices"][0].get("delta", {}).get("content", "")
                                if content:
                                    yield content
                            except json_lib.JSONDecodeError:
                                continue
            return stream_generator()
        else:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]


# ==================== RAG 引擎 ====================

class RAG:
    """RAG 检索增强生成引擎"""
    
    SYSTEM_PROMPT = """你是一个专业的知识库助手。请基于提供的参考资料回答用户问题。

要求：
1. 仅基于参考资料回答，不要编造信息
2. 如果参考资料中没有相关信息，请明确说明"根据现有资料，我无法回答这个问题"
3. 回答时请标注信息来源，格式如：[来源1]、[来源2]
4. 保持回答简洁、准确

参考资料：
{context}
"""
    
    def __init__(
        self,
        # LLM 配置
        base_url: str = None,
        model_name: str = "gpt-3.5-turbo",
        api_key: str = None,
        provider: str = None,
        # 兼容旧参数
        vllm_url: str = None,
        # Embedding 配置
        embedding_model: str = "BAAI/bge-base-zh-v1.5",
        # 存储配置
        persist_dir: str = "./faiss_db",
        # 分块配置
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        # 检索配置
        top_k: int = 5,
        # 批量处理配置
        batch_size: int = 64
    ):
        """
        初始化 RAG 引擎
        
        Args:
            base_url: LLM API 地址
            model_name: LLM 模型名称
            api_key: API 密钥
            provider: 提供商 (openai/deepseek/zhipu/moonshot/qwen/ollama/vllm)
            vllm_url: [兼容] 等同于 base_url
            embedding_model: 嵌入模型
            persist_dir: FAISS 索引存储目录
            chunk_size: 分块大小
            chunk_overlap: 分块重叠
            top_k: 默认检索数量
            batch_size: 批量 embedding 大小（默认 64，即积累 64 个 chunk 后一次性 embedding）
        
        Examples:
            # 使用 OpenAI
            rag = RAG(provider="openai", api_key="sk-xxx", model_name="gpt-4")
            
            # 使用 DeepSeek
            rag = RAG(provider="deepseek", api_key="sk-xxx", model_name="deepseek-chat")
            
            # 使用本地 vLLM
            rag = RAG(base_url="http://localhost:8000/v1", model_name="Qwen/Qwen2.5-7B")
            
            # 使用环境变量 (OPENAI_API_KEY 或 LLM_API_KEY)
            rag = RAG(provider="openai", model_name="gpt-4")
            
            # 批量添加文档（推荐）
            with RAG(...) as rag:
                for doc in documents:
                    rag.add_document(doc)
                # 退出时自动 flush
        """
        self.chunker = Chunker(chunk_size, chunk_overlap)
        self.embedder = Embedder(embedding_model)
        self.vector_store = FAISSStore(persist_dir, dim=self.embedder.dim)
        
        # 兼容旧的 vllm_url 参数
        llm_base_url = base_url or vllm_url
        
        self.llm = LLMClient(
            base_url=llm_base_url,
            model=model_name,
            api_key=api_key,
            provider=provider
        )
        self.top_k = top_k
        
        # 批量 embedding 配置
        self.batch_size = batch_size
        self._chunk_buffer: List[Chunk] = []  # 缓冲区
    
    # ==================== 批量处理方法 ====================
    
    def _add_to_buffer(self, chunks: List[Chunk]):
        """添加 chunks 到缓冲区，满了自动 flush"""
        self._chunk_buffer.extend(chunks)
        
        # 当缓冲区达到 batch_size 时自动 flush
        while len(self._chunk_buffer) >= self.batch_size:
            self._flush_batch()
    
    def _flush_batch(self):
        """flush 一批数据（batch_size 个）"""
        if not self._chunk_buffer:
            return 0
        
        # 取出一批
        batch = self._chunk_buffer[:self.batch_size]
        self._chunk_buffer = self._chunk_buffer[self.batch_size:]
        
        # 批量 embedding
        embeddings = self.embedder.embed([c.content for c in batch])
        self.vector_store.add(batch, embeddings)
        
        return len(batch)
    
    def flush(self, save: bool = True):
        """
        手动 flush 缓冲区中的所有剩余数据
        在添加完所有文档后调用，确保所有数据都被处理
        
        Args:
            save: 是否保存索引到磁盘（默认 True）
        """
        if not self._chunk_buffer:
            if save:
                self.vector_store.save()
            return 0
        
        total = 0
        # flush 所有剩余数据
        while self._chunk_buffer:
            count = self._flush_batch()
            total += count
            if count == 0:
                break
        
        # 处理不足一批的剩余数据
        if self._chunk_buffer:
            embeddings = self.embedder.embed([c.content for c in self._chunk_buffer])
            self.vector_store.add(self._chunk_buffer, embeddings)
            total += len(self._chunk_buffer)
            self._chunk_buffer = []
        
        # 保存到磁盘
        if save:
            self.vector_store.save()
        
        return total
    
    def save(self):
        """保存索引到磁盘"""
        self.vector_store.save()
    
    def get_buffer_size(self) -> int:
        """获取当前缓冲区大小"""
        return len(self._chunk_buffer)
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出时自动 flush 并保存"""
        self.flush(save=True)
        return False
    
    # ==================== 文档添加方法 ====================
    
    def add_document(self, text: str, doc_name: str = "document", auto_flush: bool = False):
        """
        添加文档到知识库
        
        Args:
            text: 文档文本
            doc_name: 文档名称
            auto_flush: 是否立即 flush（默认 False，使用批量处理）
        
        Returns:
            添加的 chunk 数量
        """
        chunks = self.chunker.chunk_text(text, doc_name)
        if not chunks:
            return 0
        
        if auto_flush:
            # 立即处理（兼容旧行为）
            embeddings = self.embedder.embed([c.content for c in chunks])
            self.vector_store.add(chunks, embeddings)
        else:
            # 加入缓冲区，批量处理
            self._add_to_buffer(chunks)
        
        return len(chunks)
    
    def add_file(self, file_path: str, auto_flush: bool = False):
        """
        添加文件到知识库
        
        Args:
            file_path: 文件路径
            auto_flush: 是否立即 flush
        """
        chunks = self.chunker.chunk_file(file_path)
        if not chunks:
            return 0
        
        if auto_flush:
            embeddings = self.embedder.embed([c.content for c in chunks])
            self.vector_store.add(chunks, embeddings)
        else:
            self._add_to_buffer(chunks)
        
        return len(chunks)
    
    def add_directory(self, dir_path: str, extensions: List[str] = None, auto_flush: bool = True):
        """
        添加目录下的所有文档
        
        Args:
            dir_path: 目录路径
            extensions: 文件扩展名列表
            auto_flush: 完成后是否自动 flush（默认 True）
        """
        extensions = extensions or [".txt", ".md", ".pdf"]
        total = 0
        for ext in extensions:
            for file_path in Path(dir_path).glob(f"**/*{ext}"):
                try:
                    count = self.add_file(str(file_path), auto_flush=False)
                    total += count
                    print(f"  {file_path.name}: {count} chunks")
                except Exception as e:
                    print(f"  Error: {file_path}: {e}")
        
        # 完成后 flush 剩余数据
        if auto_flush:
            self.flush()
        
        return total
    
    def add_chunks(self, chunks: List[Chunk], auto_flush: bool = False):
        """
        直接添加 chunks 到知识库
        
        Args:
            chunks: Chunk 列表
            auto_flush: 是否立即 flush
        """
        if not chunks:
            return 0
        
        if auto_flush:
            embeddings = self.embedder.embed([c.content for c in chunks])
            self.vector_store.add(chunks, embeddings)
        else:
            self._add_to_buffer(chunks)
        
        return len(chunks)
    
    def retrieve(self, query: str, top_k: int = None) -> List[RetrievalResult]:
        """检索相关文档"""
        top_k = top_k or self.top_k
        query_emb = self.embedder.embed_query(query)
        results = self.vector_store.search(query_emb, top_k)
        return [RetrievalResult(chunk=chunk, score=score) for chunk, score in results]
    
    def _build_context(self, results: List[RetrievalResult]) -> str:
        """构建上下文"""
        if not results:
            return "（无相关资料）"
        
        parts = []
        for i, r in enumerate(results):
            parts.append(f"[来源{i+1}] 文档: {r.chunk.doc_name}\n内容: {r.chunk.content}\n")
        return "\n---\n".join(parts)
    
    def query(self, question: str, top_k: int = None, stream: bool = False):
        """RAG查询"""
        results = self.retrieve(question, top_k)
        context = self._build_context(results)
        system_prompt = self.SYSTEM_PROMPT.format(context=context)
        
        if stream:
            return self._stream_query(question, system_prompt, results)
        else:
            answer = self.llm.generate(question, system_prompt)
            sources = [
                {"doc": r.chunk.doc_name, "score": round(r.score, 3), "content": r.chunk.content[:200]}
                for r in results
            ]
            return answer, sources
    
    def _stream_query(self, question, system_prompt, results):
        """流式查询"""
        for chunk in self.llm.generate_stream(question, system_prompt):
            yield {"type": "content", "data": chunk}
        
        sources = [
            {"doc": r.chunk.doc_name, "score": round(r.score, 3), "content": r.chunk.content[:200]}
            for r in results
        ]
        yield {"type": "sources", "data": sources}
    
    def stats(self) -> dict:
        """统计信息"""
        return {
            "total_chunks": self.vector_store.count(),
            "embedding_model": self.embedder.model_name,
            "embedding_dim": self.embedder.dim,
            "persist_dir": str(self.vector_store.persist_dir)
        }


# ==================== CLI ====================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple RAG (FAISS)")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--embedding", default="BAAI/bge-base-zh-v1.5")
    parser.add_argument("--db-dir", default="./faiss_db")
    parser.add_argument("--add-file", type=str)
    parser.add_argument("--add-dir", type=str)
    parser.add_argument("--query", "-q", type=str)
    parser.add_argument("--interactive", "-i", action="store_true")
    
    args = parser.parse_args()
    
    rag = RAG(
        vllm_url=args.vllm_url,
        model_name=args.model,
        embedding_model=args.embedding,
        persist_dir=args.db_dir
    )
    
    if args.add_file:
        print(f"Adding: {args.add_file}")
        print(f"  {rag.add_file(args.add_file)} chunks")
    
    if args.add_dir:
        print(f"Adding: {args.add_dir}")
        print(f"  Total: {rag.add_directory(args.add_dir)} chunks")
    
    if args.query:
        answer, sources = rag.query(args.query)
        print(f"\nQ: {args.query}\nA: {answer}\n")
        for i, s in enumerate(sources):
            print(f"  [{i+1}] {s['doc']} ({s['score']})")
    
    if args.interactive:
        print(f"\n{'='*50}\nRAG (FAISS) | {rag.stats()['total_chunks']} chunks\n{'='*50}\n")
        
        while True:
            try:
                q = input("You: ").strip()
                if not q: continue
                if q == 'quit': break
                if q == 'stats': print(rag.stats()); continue
                if q == 'clear': rag.vector_store.clear(); print("Cleared"); continue
                
                print("\nAssistant: ", end="", flush=True)
                sources = []
                for chunk in rag.query(q, stream=True):
                    if chunk["type"] == "content":
                        print(chunk["data"], end="", flush=True)
                    else:
                        sources = chunk["data"]
                
                print("\n\nSources:")
                for i, s in enumerate(sources):
                    print(f"  [{i+1}] {s['doc']} ({s['score']})")
                print()
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    main()