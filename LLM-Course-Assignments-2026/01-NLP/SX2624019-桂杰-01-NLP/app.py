import streamlit as st

from amrag.config import EMBED_MODEL, LLM_MODEL, PROCESS_KEYS, process_label
from amrag.ollama_client import OllamaError, check_ollama
from amrag.rag import answer_question
from amrag.store import get_collection

st.set_page_config(page_title="增材制造领域智能问答", page_icon="🧪", layout="wide")
st.title("🧪 增材制造领域智能问答系统")
st.caption("本地 Ollama 大模型 + BGE-M3 向量检索 + ChromaDB + 在线公开知识源")

with st.sidebar:
    st.header("系统设置")
    options = ["全部"] + PROCESS_KEYS
    process = st.selectbox(
        "工艺范围",
        options,
        format_func=lambda k: "全部工艺" if k == "全部" else f"{k} · {process_label(k)}",
    )
    top_k = st.slider("检索资料数量", 2, 10, 5)
    st.markdown("---")
    st.write(f"**生成模型：** `{LLM_MODEL}`")
    st.write(f"**向量模型：** `{EMBED_MODEL}`")
    st.caption(f"工艺标签共 {len(PROCESS_KEYS)} 个，可在 `amrag/processes.json` 中增删。")
    if st.button("检查本地 Ollama"):
        try:
            info = check_ollama()
            names = [m.get("name") for m in info.get("models", [])]
            st.success("Ollama 连接正常")
            st.write("已安装模型：", names)
        except OllamaError as e:
            st.error(str(e))
    try:
        st.metric("知识库记录数", get_collection().count())
    except Exception:
        st.metric("知识库记录数", "不可用")

st.info(
    "首次使用：先安装 Ollama 并执行 `ollama pull qwen2.5:7b`、`ollama pull bge-m3`，"
    "再运行 `python ingest.py --all` 建立索引。"
)

if "history" not in st.session_state:
    st.session_state.history = []

for item in st.session_state.history:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])

question = st.chat_input("例如：FDM 打印翘曲的主要原因？ / How does post-curing affect SLA resin properties?")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("正在检索本地知识库并调用本地模型……"):
            try:
                answer, sources = answer_question(question, process=process, top_k=top_k)
                st.markdown(answer)
                if sources:
                    with st.expander("查看检索来源"):
                        for i, hit in enumerate(sources, 1):
                            md = hit["metadata"]
                            st.markdown(f"**[{i}] {md.get('title') or md.get('source') or '未命名来源'}**")
                            st.caption(
                                f"工艺：{process_label(md.get('process'))} · "
                                f"类型：{md.get('source_type', 'unknown')} · "
                                f"距离：{hit['distance']:.4f}"
                            )
                            st.write((hit["text"] or "")[:1800])
                st.session_state.history.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"运行出错：{e}")
