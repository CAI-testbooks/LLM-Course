import json
import os

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")

HF_DATASET_ID = os.getenv("HF_DATASET_ID", "ppak10/Additive-Manufacturing-Benchmark")
HF_DATASET_CONFIG = os.getenv("HF_DATASET_CONFIG", "").strip()
# 默认跳过纯图像配置：它们的文本只有标签名，对问答无价值，且流式读取会传输大量图像字节。
HF_SKIP_CONFIGS = os.getenv("HF_SKIP_CONFIGS", "fdm_3d_printing_defect,machines")
OPENALEX_PER_QUERY = int(os.getenv("OPENALEX_PER_QUERY", "30"))
OPENALEX_MAILTO = os.getenv("OPENALEX_MAILTO", "").strip()

_chroma_path = os.getenv("CHROMA_PATH", "./chroma_db")
CHROMA_PATH = _chroma_path if os.path.isabs(_chroma_path) else os.path.join(BASE_DIR, _chroma_path)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "am_knowledge")

DEFAULT_PROCESS_FILE = os.path.join(BASE_DIR, "amrag", "processes.json")
PROCESS_FILE = os.getenv("PROCESS_FILE", DEFAULT_PROCESS_FILE)


def _load_process_config(path):
    """工艺标签体系由 processes.json 驱动，改文件即可增删工艺，无需改代码。"""
    if not path or not os.path.exists(path):
        path = DEFAULT_PROCESS_FILE
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    processes = [p for p in data.get("processes", []) if p.get("key")]
    if not processes:
        raise ValueError(f"{path} 中没有任何有效的工艺定义（需包含 key 字段）。")
    return processes, data.get("general_label", "General")


PROCESSES, GENERAL_LABEL = _load_process_config(PROCESS_FILE)
PROCESS_KEYS = [p["key"] for p in PROCESSES]
PROCESS_NAMES = {p["key"]: p.get("name", p["key"]) for p in PROCESSES}
PROCESS_KEYWORDS = {p["key"]: [str(k).lower() for k in p.get("keywords", [])] for p in PROCESSES}
PROCESS_PRIORITY = {p["key"]: int(p.get("priority", 0)) for p in PROCESSES}


def search_terms():
    """把所有工艺的 OpenAlex 检索式展开为 (工艺key, 检索式) 列表。"""
    terms = []
    for p in PROCESSES:
        for q in p.get("queries", []):
            q = str(q).strip()
            if q:
                terms.append((p["key"], q))
    return terms


def process_label(key):
    return PROCESS_NAMES.get(key, key)
