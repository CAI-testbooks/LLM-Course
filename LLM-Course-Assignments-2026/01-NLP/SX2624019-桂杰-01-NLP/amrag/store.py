import hashlib

import chromadb

from .config import CHROMA_PATH, COLLECTION_NAME, GENERAL_LABEL
from .ollama_client import embed_texts

_collection = None


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def stable_id(source, text):
    raw = f"{source}|{text}".encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()


def add_documents(records, batch_size=24):
    """
    records: [{text, source, metadata}, ...]
    显式传入 embeddings，避免 Chroma 再去下载自己的向量模型。
    """
    collection = get_collection()
    added = 0
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        texts = [x["text"][:12000] for x in batch]
        vectors = embed_texts(texts)
        ids, metadatas = [], []
        for item, txt in zip(batch, texts):
            ids.append(stable_id(item.get("source", "unknown"), txt))
            md = {k: str(v) for k, v in item.get("metadata", {}).items() if v is not None}
            md["source"] = str(item.get("source", "unknown"))
            md.setdefault("process", GENERAL_LABEL)
            md.setdefault("process_all", md["process"])
            metadatas.append(md)
        collection.upsert(ids=ids, documents=texts, embeddings=vectors, metadatas=metadatas)
        added += len(batch)
    return added


def _tags_of(meta):
    raw = meta.get("process_all") or meta.get("process") or GENERAL_LABEL
    return [t.strip() for t in str(raw).split(",") if t.strip()]


def search_documents(query, n_results=5, process="全部"):
    collection = get_collection()
    total = collection.count()
    if total == 0:
        return []

    qvec = embed_texts([query])[0]
    # 先多取候选，再按工艺过滤，避免过滤后结果不足 top_k。
    candidates = min(max(n_results * 4, 20), total)
    result = collection.query(
        query_embeddings=[qvec],
        n_results=candidates,
        include=["documents", "metadatas", "distances"],
    )
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    rows, fallback = [], []
    for doc, meta, distance in zip(docs, metas, distances):
        meta = meta or {}
        tags = _tags_of(meta)
        item = {"text": doc, "metadata": meta, "distance": distance}
        if process == "全部" or process in tags:
            rows.append(item)
            if process == "全部" and len(rows) >= n_results:
                break
        elif GENERAL_LABEL in tags:
            fallback.append(item)

    if len(rows) < n_results:
        rows.extend(fallback[: n_results - len(rows)])
    return rows[:n_results]
