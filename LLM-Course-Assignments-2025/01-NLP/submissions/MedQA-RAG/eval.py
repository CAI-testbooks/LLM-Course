import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from dotenv import load_dotenv

from tqdm.auto import tqdm

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from config import config

# =========================
# 1) 系统提示（只输出 answer，不再要求 citations）
# =========================

BASELINE_SYSTEM_PROMPT = """你是一个严谨的中文医疗问答助手。
目标：对用户问题给出合理、稳健的医学建议，避免编造。
要求：
1) 不要胡乱给出具体处方/剂量；可以给出常见可能原因、建议就医科室、危险信号、就医建议。
2) 如果信息不足，明确说需要哪些补充信息，并给出保守建议。
3) 输出必须是严格 JSON（不要额外文本），字段：
   - answer: string
"""

RAG_SYSTEM_PROMPT = """你是一个严谨的中文医疗问答助手。
你将收到“检索到的资料(context)”，资料来自医疗问答知识库。
目标：尽量基于资料回答；资料不足时要明确说明“不确定/资料不足”，并给出保守建议。
要求：
1) 不要编造资料中不存在的具体事实、药名剂量、检查结果等。
2) 输出必须是严格 JSON（不要额外文本），字段：
   - answer: string
"""


# =========================
# 2) 数据读取
# =========================

def load_maps(question_csv: str, answer_csv: str) -> tuple[Dict[int, str], Dict[int, str]]:
    qdf = pd.read_csv(question_csv)
    adf = pd.read_csv(answer_csv)

    q_map = {int(qid): str(content) for qid, content in zip(qdf["question_id"], qdf["content"])}
    a_map = {int(aid): str(content) for aid, content in zip(adf["ans_id"], adf["content"])}
    return q_map, a_map


def load_test_positive_samples(test_candidates_csv: str) -> pd.DataFrame:
    df = pd.read_csv(test_candidates_csv)
    df = df[df["label"] == 1].copy()
    df["question_id"] = df["question_id"].astype(int)
    df["ans_id"] = df["ans_id"].astype(int)
    return df


# =========================
# 3) 生成：baseline / rag
# =========================

def safe_json_loads(s: str) -> Dict[str, Any]:
    s = (s or "").strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s.replace("json\n", "", 1).strip()
    try:
        return json.loads(s)
    except Exception:
        return {"answer": s}


def build_baseline_chain(model):
    prompt = ChatPromptTemplate.from_messages([
        ("system", BASELINE_SYSTEM_PROMPT),
        ("human", "用户问题：{question}\n请输出 JSON："),
    ])
    return prompt | model


def build_rag_chain(model):
    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "用户问题：{question}\n\n检索到的资料(context)：\n{context}\n\n请输出 JSON："),
    ])
    return prompt | model


def docs_to_context(docs: List[Document], max_chars: int = 4000) -> str:
    blocks = []
    total = 0
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        ans_id = meta.get("answer_id", None)
        qid = meta.get("question_id", None)
        header = f"[资料{i}] answer_id={ans_id} question_id={qid}"
        body = (d.page_content or "").strip()
        block = f"{header}\n{body}\n"
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)
    return "\n".join(blocks).strip()


# =========================
# 4) Judge：准确率/幻觉率（LLM-as-judge）
# =========================

JUDGE_SYSTEM_PROMPT = """你是一个严格的自动评测器，任务是评估“模型回答”相对“参考答案/证据”的质量。
你必须只输出严格 JSON（不要额外文本），字段：
- correct: 0/1  （是否与参考答案在医学含义上基本一致、能回答问题，不要求逐字一致）
- hallucination: 0/1 （是否出现明显编造/与证据矛盾/给出证据中没有的关键医学事实如具体诊断结论、具体药物剂量等）
- rationale: string （一句话原因，尽量短）
评判标准：
1) correct=1：回答与参考答案核心含义一致或合理覆盖，且没有严重错误。
2) hallucination=1：出现关键事实无依据（相对于给定 evidence），或与 evidence 冲突，或过度具体（药物剂量/确诊结论）但 evidence 未支持。
"""


def build_judge_chain(model):
    prompt = ChatPromptTemplate.from_messages([
        ("system", JUDGE_SYSTEM_PROMPT),
        ("human",
         "用户问题：{question}\n\n"
         "模型回答：{pred_answer}\n\n"
         "evidence（参考答案或检索资料）：\n{evidence}\n\n"
         "请输出 JSON："),
    ])
    return prompt | model


# =========================
# 5) 引用F1（基于检索到的 answer_id 集合作为引用来源）
# =========================

def extract_answer_ids_from_docs(docs: List[Document]) -> Set[int]:
    """
    从检索到的 docs 中提取 metadata.answer_id，去重后作为“引用来源集合”。
    """
    out: Set[int] = set()
    for d in docs:
        meta = d.metadata or {}
        aid = meta.get("answer_id", None)
        if aid is None:
            continue
        try:
            out.add(int(aid))
        except Exception:
            pass
    return out


def citation_prf(pred_sources: Set[int], gold_ans_id: int) -> tuple[float, float, float]:
    """
    gold 只有一个：{gold_ans_id}
    precision = |pred ∩ gold| / |pred|
    recall    = |pred ∩ gold| / |gold|   (gold 大小为1)
    f1        = 2PR/(P+R)
    """
    gold = {int(gold_ans_id)}
    pred = set(int(x) for x in pred_sources) if pred_sources else set()

    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# =========================
# 6) 指标结构
# =========================

@dataclass
class EvalResult:
    correct: int
    hallucination: int
    cite_p: float
    cite_r: float
    cite_f1: float


# =========================
# 7) 主流程：baseline vs rag（进度条）
# =========================

def run_eval(
        model,
        test_df: pd.DataFrame,
        q_map: Dict[int, str],
        a_map: Dict[int, str],
        vector_store: Optional[Chroma] = None,
        top_k: int = 4,
        max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    baseline_chain = build_baseline_chain(model)
    rag_chain = build_rag_chain(model)
    judge_chain = build_judge_chain(model)

    results: List[EvalResult] = []

    eval_df = test_df if max_samples is None else test_df.head(max_samples)
    desc = "Baseline" if vector_store is None else "RAG"

    iterator = tqdm(
        eval_df.itertuples(index=False),
        total=len(eval_df),
        desc=desc,
        unit="q",
        dynamic_ncols=True
    )

    for row in iterator:
        qid = int(row.question_id)
        gold_aid = int(row.ans_id)

        question = q_map.get(qid, "")
        gold_answer = a_map.get(gold_aid, "")

        if not question or not gold_answer:
            continue

        # -------- 生成 --------
        if vector_store is None:
            resp = baseline_chain.invoke({"question": question})
            pred_obj = safe_json_loads(resp.content if hasattr(resp, "content") else str(resp))
            pred_answer = str(pred_obj.get("answer", "")).strip()
            evidence_for_judge = gold_answer

            # Baseline 没有引用来源
            pred_sources = set()

        else:
            docs = vector_store.similarity_search(question, k=top_k)
            context = docs_to_context(docs)

            resp = rag_chain.invoke({"question": question, "context": context})
            pred_obj = safe_json_loads(resp.content if hasattr(resp, "content") else str(resp))
            pred_answer = str(pred_obj.get("answer", "")).strip()
            evidence_for_judge = context

            # ✅ 引用来源：用检索到的 docs 的 answer_id 集合表示
            pred_sources = extract_answer_ids_from_docs(docs)

        # -------- 引用PRF（有效：RAG命中 gold 时 recall=1，否则0）--------
        cite_p, cite_r, cite_f1 = citation_prf(pred_sources, gold_aid)

        # -------- Judge：correct / hallucination --------
        j = judge_chain.invoke({
            "question": question,
            "pred_answer": pred_answer,
            "evidence": evidence_for_judge
        })
        j_obj = safe_json_loads(j.content if hasattr(j, "content") else str(j))

        correct = int(j_obj.get("correct", 0))
        hallucination = int(j_obj.get("hallucination", 0))

        results.append(EvalResult(
            correct=correct,
            hallucination=hallucination,
            cite_p=cite_p,
            cite_r=cite_r,
            cite_f1=cite_f1
        ))

        # 进度条显示当前均值
        if len(results) % 10 == 0:
            acc_now = sum(r.correct for r in results) / len(results)
            hallu_now = sum(r.hallucination for r in results) / len(results)
            citef1_now = sum(r.cite_f1 for r in results) / len(results)
            iterator.set_postfix(acc=f"{acc_now:.3f}", hallu=f"{hallu_now:.3f}", citeF1=f"{citef1_now:.3f}")

    if not results:
        return {"n": 0}

    acc = sum(r.correct for r in results) / len(results)
    hallu = sum(r.hallucination for r in results) / len(results)
    cite_p = sum(r.cite_p for r in results) / len(results)
    cite_r = sum(r.cite_r for r in results) / len(results)
    cite_f1 = sum(r.cite_f1 for r in results) / len(results)

    return {
        "n": len(results),
        "accuracy": acc,
        "hallucination_rate": hallu,
        "citation_precision": cite_p,
        "citation_recall": cite_r,
        "citation_f1": cite_f1,
    }


def main():
    load_dotenv()

    model = init_chat_model(
        model="ollama:deepseek-r1:8b",
        base_url="http://localhost:11434",
        temperature=0.1,
    )

    question_csv = config["question_dict"]
    answer_csv = config["answer_dict"]
    test_candidates_csv = config["test_dict"]

    q_map, a_map = load_maps(question_csv, answer_csv)
    test_df = load_test_positive_samples(test_candidates_csv)

    MAX_SAMPLES = config.get("samples_num", None)

    print(f"Total positive samples: {len(test_df)}")
    print(f"Eval samples: {MAX_SAMPLES if MAX_SAMPLES is not None else 'ALL'}")

    # -------- Baseline --------
    print("\n=== Baseline (No RAG) ===")
    baseline_metrics = run_eval(
        model=model,
        test_df=test_df,
        q_map=q_map,
        a_map=a_map,
        vector_store=None,
        top_k=4,
        max_samples=MAX_SAMPLES,
    )
    print(json.dumps(baseline_metrics, ensure_ascii=False, indent=2))

    # -------- RAG --------
    print("\n=== RAG (Chroma + OllamaEmbeddings) ===")

    embedding = OllamaEmbeddings(
        model="qwen3-embedding:4b",
        base_url="http://localhost:11434",
    )

    vector_store = Chroma(
        collection_name="medical_rag_collection",
        embedding_function=embedding,
        persist_directory="./chroma_rag_db"
    )

    rag_metrics = run_eval(
        model=model,
        test_df=test_df,
        q_map=q_map,
        a_map=a_map,
        vector_store=vector_store,
        top_k=4,
        max_samples=MAX_SAMPLES,
    )
    print(json.dumps(rag_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
