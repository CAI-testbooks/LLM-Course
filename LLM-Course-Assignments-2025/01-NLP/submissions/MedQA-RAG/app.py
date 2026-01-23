import os
import re
from typing import List, Dict, Any, Tuple, Optional

import streamlit as st
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# =========================
# 基础配置
# =========================
load_dotenv()

st.set_page_config(page_title="Medical RAG Demo", page_icon="🩺", layout="wide")


# =========================
# 你的模型 / Embedding / 向量库初始化方式（严格按你的写法）
# =========================
@st.cache_resource
def load_models_and_store():
    model = init_chat_model(
        model="ollama:deepseek-r1:8b",
        base_url="http://localhost:11434",
        temperature=0.1,
    )

    embedding = OllamaEmbeddings(
        model="qwen3-embedding:4b",
        base_url="http://localhost:11434",
    )

    vector_store = Chroma(
        collection_name="medical_rag_collection",
        embedding_function=embedding,
        persist_directory="./chroma_rag_db"
    )

    return model, embedding, vector_store


model, embedding, vector_store = load_models_and_store()

# =========================
# Prompts（拒绝不确定 + 引用来源 + 多轮）
# =========================

# 1) 对话摘要（长对话用）
SUMMARY_SYSTEM = """你是一个对话记录压缩器。
将对话历史压缩成“事实摘要”，保留：用户症状/时间线/关键检查/关键结论/未解决问题/重要限制条件。
不要编造。输出中文纯文本，尽量短（<= 600字）。"""

summary_prompt = ChatPromptTemplate.from_messages([
    ("system", SUMMARY_SYSTEM),
    ("human", "已有摘要：\n{summary}\n\n新增对话片段：\n{new_lines}\n\n请输出更新后的摘要：")
])

# 2) 检索 query 重写（更适合多轮）
REWRITE_SYSTEM = """你是一个查询改写器。
给定对话摘要 + 最近对话，把用户最新问题改写成“独立、可检索”的中文查询（只输出一行查询，不要解释）。"""

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", REWRITE_SYSTEM),
    ("human", "对话摘要：{summary}\n\n最近对话：\n{recent}\n\n用户最新问题：{question}\n\n独立检索查询：")
])

# 3) RAG 回答（拒答 + 引用）
ANSWER_SYSTEM = """你是一个严谨的中文医疗问答助手（RAG）。
你会收到：对话摘要、最近对话、以及检索资料(context)。
要求：
1) 只能依据 context 中的信息回答；如果 context 不足以支持结论，必须明确说“资料不足/不确定”，并给出保守建议（如建议就医科室、危险信号、需要补充的信息）。
2) 严禁编造：不要给出 context 中没有的关键事实（尤其是确诊结论、具体药名剂量、检查结果）。
3) 必须输出 Markdown，并包含两个部分：
   - 【回答】...（面向用户）
   - 【引用来源】列出你用到的资料条目编号（如 1、2）以及对应的 answer_id（来自资料头部 answer_id=...）
"""

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", ANSWER_SYSTEM),
    ("human",
     "对话摘要：\n{summary}\n\n"
     "最近对话：\n{recent}\n\n"
     "用户问题：{question}\n\n"
     "检索资料(context)：\n{context}\n\n"
     "请回答：")
])


# =========================
# 工具函数
# =========================
def format_recent_messages(messages: List[Dict[str, str]], max_turns: int = 8) -> str:
    """
    保留最近 max_turns 轮（user+assistant 算一轮）原文，用于增强连贯性。
    """
    # messages: [{"role": "user"/"assistant", "content": "..."}]
    if not messages:
        return ""
    # 取最后 2*max_turns 条消息
    tail = messages[-2 * max_turns:]
    lines = []
    for m in tail:
        role = "用户" if m["role"] == "user" else "助手"
        lines.append(f"{role}：{m['content']}")
    return "\n".join(lines)


def docs_to_context_with_ids(docs_with_scores: List[Tuple[Document, float]], max_chars: int = 4500) -> str:
    """
    把检索资料拼成 context，并显式标号 + answer_id，方便“引用来源”输出。
    score 仅用于展示，不要求模型理解。
    """
    blocks = []
    total = 0
    for idx, (doc, score) in enumerate(docs_with_scores, start=1):
        meta = doc.metadata or {}
        ans_id = meta.get("answer_id", None)
        qid = meta.get("question_id", None)
        chunk_id = meta.get("chunk_id", None)
        header = f"[资料{idx}] answer_id={ans_id} question_id={qid} chunk_id={chunk_id} score={score:.4f}"
        body = (doc.page_content or "").strip()
        block = f"{header}\n{body}\n"
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)
    return "\n".join(blocks).strip()


def extract_used_source_nums(answer_md: str) -> List[int]:
    """
    从【引用来源】里粗略抽取“资料编号”（1、2、3...）。
    不是必须准确，只用于 UI 高亮（失败也无所谓）。
    """
    if not answer_md:
        return []
    # 匹配“资料1 / 1 / 1、2”等
    nums = re.findall(r"(?:资料)?\s*([1-9]\d*)", answer_md)
    out = []
    for n in nums:
        try:
            out.append(int(n))
        except:
            pass
    # 去重保序
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def should_refuse_by_score(docs_with_scores: List[Tuple[Document, float]], distance_threshold: float) -> bool:
    """
    Chroma 的 similarity_search_with_score 返回的 score 常见是“距离”，越小越相近。
    若最相近的距离仍然很大 => 检索不可靠 => 触发拒答。
    """
    if not docs_with_scores:
        return True
    best = docs_with_scores[0][1]
    return best > distance_threshold


# =========================
# Streamlit UI
# =========================
st.title("🩺 医疗领域 RAG 问答 Demo（多轮 + 引用 + 拒答）")

with st.sidebar:
    st.header("参数")
    k = st.slider("检索 Top-K", min_value=1, max_value=10, value=4, step=1)
    # 这个阈值需要你根据 embedding/库分布调一下。默认给个保守值。
    distance_threshold = st.slider("拒答距离阈值（越小越严格）", min_value=0.1, max_value=2.0, value=0.8, step=0.05)
    max_turns = st.slider("保留最近对话轮数", min_value=2, max_value=20, value=8, step=1)
    enable_query_rewrite = st.checkbox("启用多轮检索查询改写", value=True)
    st.divider()
    if st.button("🧹 清空对话"):
        st.session_state.messages = []
        st.session_state.summary = ""
        st.session_state.last_sources = []
        st.rerun()

# 会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": ..., "content": ...}
if "summary" not in st.session_state:
    st.session_state.summary = ""  # 滚动摘要
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []  # 最近一次检索到的 docs_with_scores

# 展示历史消息
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# 输入框
user_text = st.chat_input("请输入医疗问题（支持多轮对话）…")

if user_text:
    # 记录用户消息
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # 生成 recent + summary
    recent_text = format_recent_messages(st.session_state.messages, max_turns=max_turns)
    summary_text = st.session_state.summary

    # 1) 查询改写（可选）
    if enable_query_rewrite:
        rewrite_chain = rewrite_prompt | model
        rewrite_resp = rewrite_chain.invoke({
            "summary": summary_text,
            "recent": recent_text,
            "question": user_text
        })
        query = rewrite_resp.content.strip() if hasattr(rewrite_resp, "content") else str(rewrite_resp).strip()
    else:
        query = user_text.strip()

    # 2) 检索（带 score）
    docs_with_scores = vector_store.similarity_search_with_score(query, k=k)
    st.session_state.last_sources = docs_with_scores
    context_text = docs_to_context_with_ids(docs_with_scores)

    # 3) 拒答判定（检索不可靠 -> 拒绝）
    refuse = should_refuse_by_score(docs_with_scores, distance_threshold=distance_threshold)

    # 4) 生成回答（流式）
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_answer = ""

        if refuse:
            # 直接拒答（仍给出安全建议）
            full_answer = (
                "【回答】\n"
                "我在当前知识库中没有检索到足够可靠的依据来回答这个问题（资料相似度不足）。\n\n"
                "你可以补充：症状持续时间、是否发热/咳嗽、既往病史、用药史、检查结果等；\n"
                "如果出现胸痛、呼吸困难、持续高热、意识改变等危险信号，请尽快就医。\n\n"
                "【引用来源】\n"
                "（无：本次检索结果不可靠，未引用）"
            )
            placeholder.markdown(full_answer)
        else:
            answer_chain = answer_prompt | model
            # Streamlit 实时输出
            for chunk in answer_chain.stream({
                "summary": summary_text,
                "recent": recent_text,
                "question": user_text,
                "context": context_text
            }):
                # chunk 可能是 AIMessageChunk
                part = chunk.content if hasattr(chunk, "content") else str(chunk)
                full_answer += part
                placeholder.markdown(full_answer)

    # 5) 记录助手消息
    st.session_state.messages.append({"role": "assistant", "content": full_answer})

    # 6) 更新滚动摘要（当消息很长时把旧对话压缩进 summary）
    # 简单策略：每次都用“最近一段对话”更新摘要（你也可改成每 N 轮更新一次）
    try:
        # 取最后几条用于“新增片段”
        new_lines = format_recent_messages(st.session_state.messages, max_turns=min(max_turns, 6))
        sum_chain = summary_prompt | model
        sum_resp = sum_chain.invoke({"summary": st.session_state.summary, "new_lines": new_lines})
        st.session_state.summary = sum_resp.content.strip() if hasattr(sum_resp, "content") else str(sum_resp).strip()
    except Exception:
        # 摘要失败不影响主流程
        pass

    # 7) 显示引用来源（UI 展示检索到的资料）
    st.divider()
    st.subheader("📌 本轮检索到的资料（可解释引用）")
    used_nums = extract_used_source_nums(full_answer)

    for idx, (doc, score) in enumerate(docs_with_scores, start=1):
        meta = doc.metadata or {}
        title = f"资料{idx} | answer_id={meta.get('answer_id')} | question_id={meta.get('question_id')} | score={score:.4f}"
        if idx in used_nums:
            title = "✅ " + title

        with st.expander(title, expanded=(idx == 1)):
            st.markdown(doc.page_content)
            st.caption(f"metadata: {meta}")
