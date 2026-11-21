import requests

from .config import EMBED_MODEL, LLM_MODEL, OLLAMA_BASE_URL


class OllamaError(RuntimeError):
    pass


def check_ollama():
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=8)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise OllamaError(
            "无法连接 Ollama："
            f"{OLLAMA_BASE_URL}。请先安装并启动 Ollama（https://ollama.com/download），"
            f"再在终端执行 ollama list 确认服务可用。详情：{e}"
        ) from e


def embed_texts(texts, model=None):
    """调用 Ollama 本地 /api/embed，返回 list[list[float]]。"""
    if not texts:
        return []
    model = model or EMBED_MODEL
    payload = {"model": model, "input": texts}
    try:
        r = requests.post(f"{OLLAMA_BASE_URL}/api/embed", json=payload, timeout=180)
        r.raise_for_status()
        data = r.json()
        vectors = data.get("embeddings")
        if not vectors:
            raise OllamaError(f"Ollama embedding 响应中没有 embeddings：{data}")
        return vectors
    except requests.RequestException as e:
        raise OllamaError(
            f"向量模型调用失败。请确认已执行：ollama pull {model}。详情：{e}"
        ) from e


def chat(messages, model=None, temperature=0.2):
    model = model or LLM_MODEL
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    try:
        r = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=300)
        r.raise_for_status()
        return r.json()["message"]["content"]
    except requests.RequestException as e:
        raise OllamaError(
            f"本地大模型调用失败。请确认已执行：ollama pull {model}。详情：{e}"
        ) from e
