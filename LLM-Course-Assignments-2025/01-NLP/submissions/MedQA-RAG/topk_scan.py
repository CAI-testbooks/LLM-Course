import os
import csv
import json
import argparse
import inspect
from typing import List, Dict, Any, Optional

import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# 直接复用你现有 eval.py
import eval as eval_mod
from config import config


def parse_topks(s: str) -> List[int]:
    vals = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    # 去重并保序
    seen = set()
    out = []
    for x in vals:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def safe_get(d: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        v = d.get(key, default)
        return float(v)
    except Exception:
        return float(default)


def call_run_eval(
    *,
    model,
    test_df,
    q_map,
    a_map,
    vector_store: Optional[Chroma],
    top_k: int,
    max_samples: Optional[int],
    citation_fallback_mode: Optional[str],
) -> Dict[str, Any]:
    """
    兼容不同版本 run_eval 的签名：
    """
    sig = inspect.signature(eval_mod.run_eval)
    kwargs = dict(
        model=model,
        test_df=test_df,
        q_map=q_map,
        a_map=a_map,
        vector_store=vector_store,
        top_k=top_k,
        max_samples=max_samples,
    )
    if "citation_fallback_mode" in sig.parameters and citation_fallback_mode is not None:
        kwargs["citation_fallback_mode"] = citation_fallback_mode
    return eval_mod.run_eval(**kwargs)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_markdown_table(df: pd.DataFrame, path: str) -> None:
    md = df.to_markdown(index=False)
    with open(path, "w", encoding="utf-8") as f:
        f.write(md + "\n")


def plot_curve(xs: List[int], ys: List[float], title: str, xlabel: str, ylabel: str, outpath: str) -> None:
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(xs)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Top-K scan for RAG evaluation (generate table + curves).")
    parser.add_argument("--topks", type=str, default="1,2,4,8", help="Comma-separated top-k list, e.g. 1,2,4,8")
    parser.add_argument("--max_samples", type=int, default=None, help="Override config['samples_num']; None means use config")
    parser.add_argument("--outdir", type=str, default="results", help="Output directory for csv/md/png")
    parser.add_argument("--collection", type=str, default="medical_rag_collection", help="Chroma collection name")
    parser.add_argument("--persist_dir", type=str, default="./chroma_rag_db", help="Chroma persist directory")
    args = parser.parse_args()

    load_dotenv()
    ensure_dir(args.outdir)

    # ====== 1) Model ======
    model = init_chat_model(
        model="ollama:deepseek-r1:8b",
        base_url="http://localhost:11434",
        temperature=0.1,
    )

    # ====== 2) Load data ======
    question_csv = config["question_dict"]
    answer_csv = config["answer_dict"]
    test_candidates_csv = config["test_dict"]

    q_map, a_map = eval_mod.load_maps(question_csv, answer_csv)
    test_df = eval_mod.load_test_positive_samples(test_candidates_csv)

    max_samples = args.max_samples if args.max_samples is not None else config.get("samples_num", None)

    # 如果你 eval.py 里支持这个配置，就沿用；否则忽略
    citation_fallback_mode = config.get("citation_fallback_mode", None)

    print(f"Total test positive samples: {len(test_df)}")
    print(f"Eval samples: {max_samples if max_samples is not None else 'ALL'}")

    # ====== 3) Build vector store once ======
    embedding = OllamaEmbeddings(
        model="qwen3-embedding:4b",
        base_url="http://localhost:11434",
    )
    vector_store = Chroma(
        collection_name=args.collection,
        embedding_function=embedding,
        persist_directory=args.persist_dir,
    )

    # ====== 4) Baseline (optional but useful for table) ======
    print("\n=== Baseline (No RAG) ===")
    baseline_metrics = call_run_eval(
        model=model,
        test_df=test_df,
        q_map=q_map,
        a_map=a_map,
        vector_store=None,
        top_k=4,  # baseline 不用检索，top_k 无意义，这里随便给个值
        max_samples=max_samples,
        citation_fallback_mode=citation_fallback_mode,
    )
    print(json.dumps(baseline_metrics, ensure_ascii=False, indent=2))

    # ====== 5) Top-K scan (RAG) ======
    topks = parse_topks(args.topks)
    rows: List[Dict[str, Any]] = []

    # 先把 baseline 放进表格
    rows.append({
        "method": "Baseline",
        "top_k": "-",
        "n": baseline_metrics.get("n", 0),
        "accuracy": safe_get(baseline_metrics, "accuracy"),
        "hallucination_rate": safe_get(baseline_metrics, "hallucination_rate"),
        "citation_f1": safe_get(baseline_metrics, "citation_f1"),
    })

    for k in topks:
        print(f"\n=== RAG Top-K={k} ===")
        m = call_run_eval(
            model=model,
            test_df=test_df,
            q_map=q_map,
            a_map=a_map,
            vector_store=vector_store,
            top_k=k,
            max_samples=max_samples,
            citation_fallback_mode=citation_fallback_mode,
        )
        print(json.dumps(m, ensure_ascii=False, indent=2))

        rows.append({
            "method": "RAG",
            "top_k": k,
            "n": m.get("n", 0),
            "accuracy": safe_get(m, "accuracy"),
            "hallucination_rate": safe_get(m, "hallucination_rate"),
            "citation_f1": safe_get(m, "citation_f1"),
        })

    # ====== 6) Save table ======
    df = pd.DataFrame(rows)

    csv_path = os.path.join(args.outdir, "topk_scan.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    md_path = os.path.join(args.outdir, "topk_scan.md")
    save_markdown_table(df, md_path)

    print(f"\nSaved table:")
    print(f"- {csv_path}")
    print(f"- {md_path}")

    # ====== 7) Plot curves (RAG rows only) ======
    rag_df = df[df["method"] == "RAG"].copy()
    if len(rag_df) > 0:
        xs = [int(x) for x in rag_df["top_k"].tolist()]
        accs = rag_df["accuracy"].astype(float).tolist()
        cfs = rag_df["citation_f1"].astype(float).tolist()
        halls = rag_df["hallucination_rate"].astype(float).tolist()

        plot_curve(xs, accs, "Top-K vs Accuracy (RAG)", "Top-K", "Accuracy", os.path.join(args.outdir, "topk_accuracy.png"))
        plot_curve(xs, cfs, "Top-K vs Citation F1 (RAG)", "Top-K", "Citation F1", os.path.join(args.outdir, "topk_citation_f1.png"))
        plot_curve(xs, halls, "Top-K vs Hallucination Rate (RAG)", "Top-K", "Hallucination Rate", os.path.join(args.outdir, "topk_hallucination_rate.png"))

        print(f"\nSaved plots in: {args.outdir}/")
        print("- topk_accuracy.png")
        print("- topk_citation_f1.png")
        print("- topk_hallucination_rate.png")
    else:
        print("\nNo RAG rows found; skip plotting.")

    # 额外：把表格打印到终端（方便你直接复制到报告）
    print("\n=== Markdown Table Preview ===")
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
