import time

import requests
from datasets import get_dataset_config_names, load_dataset

from .config import (
    GENERAL_LABEL,
    HF_DATASET_CONFIG,
    HF_DATASET_ID,
    HF_SKIP_CONFIGS,
    OPENALEX_MAILTO,
    OPENALEX_PER_QUERY,
    PROCESS_KEYWORDS,
    PROCESS_PRIORITY,
    search_terms,
)


def infer_process(text):
    """按 processes.json 的关键词打分，返回 (主工艺标签, 命中标签列表)。"""
    lowered = text.lower()
    scores = {}
    for key, keywords in PROCESS_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw and kw in lowered)
        if hits:
            scores[key] = hits
    if not scores:
        return GENERAL_LABEL, [GENERAL_LABEL]
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], -PROCESS_PRIORITY.get(kv[0], 0)))
    return ordered[0][0], [key for key, _ in ordered]


def _flatten(value):
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        return " ".join(f"{k}: {_flatten(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return " ".join(_flatten(v) for v in value)
    return ""


def _row_to_text(row):
    """整理成可检索的知识文本：问答类保留题干/选项/答案/解释，其余保留文本字段。"""
    lower = {str(k).lower(): k for k in row.keys()}

    def pick(*names):
        for n in names:
            if n in lower:
                return _flatten(row[lower[n]]).strip()
        return ""

    question = pick("question", "query", "instruction", "prompt")
    if question:
        choices = pick("choices", "options")
        correct = pick("correct_answer", "correct")
        answer = pick("answer", "answer_solution", "response", "output", "completion")
        explanation = pick("explanation", "rationale", "rubric")
        parts = [f"Question: {question}"]
        if choices:
            parts.append(f"Options: {choices}")
        if correct:
            parts.append(f"Correct answer: {correct}")
        if answer and answer != correct:
            parts.append(f"Answer: {answer}")
        if explanation:
            parts.append(f"Explanation: {explanation}")
        if len(parts) > 1:
            return "\n".join(parts)

    useful = []
    for k, v in row.items():
        s = _flatten(v).strip()
        if s and str(k).lower() not in ("image", "audio", "video", "image_ext"):
            useful.append(f"{k}: {s}")
    return "\n".join(useful)


def stream_huggingface(limit_per_config=500):
    """
    流式读取公开 Hugging Face 数据集。streaming=True 不会先下载完整数据集，
    但读取的数据仍会通过网络传输；datasets 库可能缓存少量元数据。
    """
    dataset_id = HF_DATASET_ID
    if HF_DATASET_CONFIG:
        configs = [HF_DATASET_CONFIG]
    else:
        try:
            available = get_dataset_config_names(dataset_id)
            skip = {s.strip() for s in HF_SKIP_CONFIGS.split(",") if s.strip()}
            # 采集全部非图像配置（可用 HF_SKIP_CONFIGS 排除），避免漏掉特定工艺的数据。
            configs = [c for c in available if c not in skip] or list(available)
        except Exception:
            configs = [None]

    for config in configs:
        try:
            kwargs = {"split": "train", "streaming": True}
            if config:
                kwargs["name"] = config
            stream = load_dataset(dataset_id, **kwargs)
            count = 0
            for row in stream:
                text = _row_to_text(row)
                if len(text.strip()) < 30:
                    continue
                primary, tags = infer_process(text)
                yield {
                    "text": f"Dataset: {dataset_id}\nConfig: {config or 'default'}\n{text}",
                    "source": f"Hugging Face: {dataset_id}",
                    "metadata": {
                        "process": primary,
                        "process_all": ",".join(tags),
                        "source_type": "dataset",
                        "config": config or "default",
                    },
                }
                count += 1
                if count >= limit_per_config:
                    break
        except Exception as e:
            print(f"[WARN] HF 配置 {config!r} 读取失败：{e}")


def _abstract_from_inverted_index(index):
    if not index:
        return ""
    words = {}
    for word, positions in index.items():
        for pos in positions:
            words[int(pos)] = word
    return " ".join(words[i] for i in sorted(words))


def search_openalex(per_query=None, only_processes=None):
    """从 OpenAlex 获取论文元数据与摘要；不下载论文全文。"""
    per_query = per_query or OPENALEX_PER_QUERY
    headers = {"User-Agent": f"AM-RAG/1.0 (mailto:{OPENALEX_MAILTO})"} if OPENALEX_MAILTO else {}
    allowed = set(only_processes) if only_processes else None

    for process, query in search_terms():
        if allowed and process not in allowed:
            continue
        params = {
            "search": query,
            "per-page": min(int(per_query), 100),
            "select": "id,doi,display_name,publication_year,abstract_inverted_index,primary_location",
        }
        try:
            r = requests.get("https://api.openalex.org/works", params=params, headers=headers, timeout=45)
            r.raise_for_status()
            for item in r.json().get("results", []):
                title = item.get("display_name") or ""
                abstract = _abstract_from_inverted_index(item.get("abstract_inverted_index"))
                if len(abstract) < 80:
                    continue
                doi = item.get("doi") or item.get("id") or ""
                year = item.get("publication_year") or ""
                text = f"Title: {title}\nYear: {year}\nDOI/URL: {doi}\nAbstract: {abstract}"
                yield {
                    "text": text,
                    "source": doi,
                    "metadata": {
                        "process": process,
                        "process_all": process,
                        "source_type": "paper_abstract",
                        "title": title,
                        "year": year,
                        "doi": doi,
                    },
                }
            time.sleep(0.3)
        except Exception as e:
            print(f"[WARN] OpenAlex 查询失败（{process}）：{e}")


def collect_online(hf_limit=300, openalex_per_query=None, use_hf=True, use_openalex=True, only_processes=None):
    records = []
    if use_hf:
        records.extend(list(stream_huggingface(limit_per_config=hf_limit)))
    if use_openalex:
        records.extend(list(search_openalex(openalex_per_query, only_processes=only_processes)))

    # 去掉本次运行内的重复文本。
    seen, unique = set(), []
    for r in records:
        key = (r["source"], r["text"][:500])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique
