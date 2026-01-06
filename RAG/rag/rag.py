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
    
    def _save(self):
        """保存索引"""
        import faiss
        faiss.write_index(self.index, str(self.index_path))
        with open(self.chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
    
    def add(self, chunks: List[Chunk], embeddings: np.ndarray):
        """添加文档块"""
        if len(chunks) == 0:
            return
        
        embeddings = embeddings.astype('float32')
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        self._save()
        
        print(f"Added {len(chunks)} chunks (total: {len(self.chunks)})")
    
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
        if self.index_path.exists():
            self.index_path.unlink()
        if self.chunks_path.exists():
            self.chunks_path.unlink()


# ==================== LLM 客户端 ====================

class LLMClient:
    """vLLM OpenAI兼容客户端"""
    
    def __init__(
        self, 
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen2.5-7B-Instruct"
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
    
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
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            },
            timeout=120.0
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
        import json
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        with httpx.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True
            },
            timeout=120.0
        ) as response:
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        content = chunk["choices"][0].get("delta", {}).get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue


# ==================== RAG 引擎 ====================

class RAG:
    """RAG"""
    
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
        vllm_url: str = "http://localhost:8000/v1",
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        embedding_model: str = "BAAI/bge-base-zh-v1.5",
        persist_dir: str = "./faiss_db",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k: int = 5
    ):
        self.chunker = Chunker(chunk_size, chunk_overlap)
        self.embedder = Embedder(embedding_model)
        self.vector_store = FAISSStore(persist_dir, dim=self.embedder.dim)
        self.llm = LLMClient(vllm_url, model_name)
        self.top_k = top_k
    
    def add_document(self, text: str, doc_name: str = "document"):
        """添加文档到知识库"""
        chunks = self.chunker.chunk_text(text, doc_name)
        if chunks:
            embeddings = self.embedder.embed([c.content for c in chunks])
            self.vector_store.add(chunks, embeddings)
        return len(chunks)
    
    def add_file(self, file_path: str):
        """添加文件到知识库"""
        chunks = self.chunker.chunk_file(file_path)
        if chunks:
            embeddings = self.embedder.embed([c.content for c in chunks])
            self.vector_store.add(chunks, embeddings)
        return len(chunks)
    
    def add_directory(self, dir_path: str, extensions: List[str] = None):
        """添加目录下的所有文档"""
        extensions = extensions or [".txt", ".md", ".pdf"]
        total = 0
        for ext in extensions:
            for file_path in Path(dir_path).glob(f"**/*{ext}"):
                try:
                    count = self.add_file(str(file_path))
                    total += count
                    print(f"  {file_path.name}: {count} chunks")
                except Exception as e:
                    print(f"  Error: {file_path}: {e}")
        return total
    
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