import streamlit as st
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import time
import torch
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSequenceClassification
import gc

# ==========================================
# 环境与常量
# ==========================================

ST_TITLE = "通用中文医疗领域智能问答系统"
#MODEL_NAME = "/root/autodl-tmp/qwen/Qwen2___5-7B-Instruct"
MODEL_NAME = "/root/autodl-tmp/Medical-RAG/Tune-model/medical-qwen-merged"  # 修改为merage后的模型路径
EMBEDDING_MODEL = "BAAI/bge-m3"
VECTOR_DB_PATH = "/root/autodl-tmp/Medical-RAG/chroma_db_medical"
DATASET_DIR = "/root/autodl-tmp/Medical-RAG/dataset"

# 检索与重排序参数
MMR_FETCH_K = 15
RERANKER_RETRIEVE_K = 10
RERANKER_TOP_K = 3
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
SCORE_THRESHOLD = 2.0

# ==========================================
# 工具函数
# ==========================================

def load_alpaca_json_as_documents(file_path):
    docs = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)
        for idx, item in enumerate(data_list, 1):
            instruction = item.get("instruction", "").strip()
            output = item.get("output", "").strip()
            if not instruction or not output:
                continue
            page_content = f"问题：{instruction}\n答案：{output}"
            metadata = {
                "source_instruction": instruction,
                "source_file": os.path.basename(file_path),
                "item_index": idx,
            }
            docs.append(Document(page_content=page_content, metadata=metadata))
    except Exception as e:
        st.error(f"加载 {file_path} 出错: {e}")
        return []
    return docs


def normalize_messages(messages):
    """将 messages 转换为标准格式列表，过滤掉非法项。"""
    if not isinstance(messages, (list, tuple)):
        return []
    normalized = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            normalized.append({"role": role, "content": content.strip()})
    return normalized


def truncate_chat_history(messages, tokenizer, max_tokens=1000):
    """按 token 截断对话历史（从最新往回取）"""
    messages = normalize_messages(messages)
    if not messages:
        return ""

    formatted_lines = []
    total_tokens = 0
    for msg in reversed(messages):
        role_name = "患者" if msg["role"] == "user" else "医生"
        line = f"{role_name}：{msg['content']}"
        tokens = len(tokenizer.encode(line, add_special_tokens=False))
        if total_tokens + tokens > max_tokens:
            break
        formatted_lines.append(line)
        total_tokens += tokens
    return "\n".join(reversed(formatted_lines))


# ==========================================
# 增强版 BGE-Reranker
# ==========================================
class BGERReranker:
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3", device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
        self.device = device

    def rerank(self, query: str, documents: List[Document], top_k: int = 3, threshold: float = SCORE_THRESHOLD):
        if not documents:
            return [], True
        
        pairs = [[query, doc.page_content] for doc in documents]
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            ).to(self.device)
            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()

        scored_docs = list(zip(documents, scores.cpu().numpy()))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        if scored_docs and scored_docs[0][1] < threshold:
            return [], True

        top_docs = [doc for doc, _ in scored_docs[:top_k]]
        return top_docs, False


# ==========================================
# 初始化 RAG 系统
# ==========================================
@st.cache_resource
def initialize_rag_system():
    if not os.path.exists(DATASET_DIR):
        return None, f"找不到数据集目录: {DATASET_DIR}"

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}
    )

    if os.path.exists(VECTOR_DB_PATH):
        vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
    else:
        json_files = [
            os.path.join(DATASET_DIR, "alpaca_formatted_test_data.json"),
            os.path.join(DATASET_DIR, "alpaca_formatted_validation_data.json"),
            os.path.join(DATASET_DIR, "alpaca_formatted_train_data.json")
        ]
        docs = []
        for file_path in json_files:
            if os.path.exists(file_path):
                docs.extend(load_alpaca_json_as_documents(file_path))
        if not docs:
            return None, "未加载到任何有效文档"
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        splits = splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=VECTOR_DB_PATH
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    def get_mmr_lambda(query: str) -> float:
        high_risk_keywords = ["急", "立即", "急救", "用药", "剂量", "手术", "过敏", "死亡"]
        if any(kw in query for kw in high_risk_keywords):
            return 0.95
        else:
            return 0.6

    def mmr_retrieve_with_dynamic_lambda(query: str):
        lambda_mult = get_mmr_lambda(query)
        search_kwargs = {
            "k": RERANKER_RETRIEVE_K,
            "fetch_k": MMR_FETCH_K,
            "lambda_mult": lambda_mult
        }
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs=search_kwargs)
        return retriever.invoke(query)

    reranker = BGERReranker(model_name=RERANKER_MODEL, device="cuda")

    def retrieve_and_rerank(inputs: Dict[str, Any]) -> str:
        query = inputs["question"]
        chat_history_str = inputs.get("chat_history", "")
        context_query = f"{chat_history_str}\n当前问题：{query}".strip() if chat_history_str else query

        docs = mmr_retrieve_with_dynamic_lambda(query)
        top_docs, is_unanswerable = reranker.rerank(context_query, docs, top_k=RERANKER_TOP_K)
        
        if is_unanswerable or not top_docs:
            return "根据现有医学资料，我无法提供确切答案，建议咨询专业医生。"
        
        return "\n\n".join([f"【参考片段 {i+1}】\n{doc.page_content}" for i, doc in enumerate(top_docs)])

    retriever_runnable = RunnableLambda(retrieve_and_rerank)

    # LLM 初始化
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.95,
        repetition_penalty=1.1,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        clean_up_tokenization_spaces=True,
        early_stopping=True
    )
    from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
    llm_pipeline = HuggingFacePipeline(pipeline=pipe)
    llm = ChatHuggingFace(llm=llm_pipeline, tokenizer=tokenizer, streaming=True)

    template = """
你是一个严谨的中文医学专家助手。
请严格基于以下“医学知识”作答，不得编造、推测或引入外部知识。
若医学知识中无相关信息，请直接回答：“根据现有医学资料，我无法提供确切答案，建议咨询专业医生。”

回答要求：
1. 若涉及急症、用药、手术、过敏等高风险内容，开头必须加：
   “⚠️ 注意：此信息不能替代紧急医疗救助，请立即联系医生或前往医院。”
2. 回答需条理清晰，使用有序列表（1. 2. 3.）或无序列表（- ...）组织内容；
3. 每个观点只表达一次，禁止冗余；
4. 使用规范医学术语（如“心肌梗死”而非“心梗”）；
5. 若信息不确定，请使用“可能”“部分研究表明”等措辞。

以下是最近的对话历史（如有）：
{chat_history}

医学知识：
{context}

当前用户问题：
{question}
"""

    prompt = ChatPromptTemplate.from_template(template)

    # ✅ 修正：只从 input_data 字典中提取 messages
    def format_chat_history_for_prompt(input_data: Dict[str, Any]) -> str:
        messages = input_data.get("messages", [])
        return truncate_chat_history(messages, tokenizer, max_tokens=800)

    rag_chain = (
        {
            "context": retriever_runnable,
            "chat_history": RunnableLambda(format_chat_history_for_prompt),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain._tokenizer = tokenizer
    rag_chain._model = model
    return rag_chain, "成功"


# ==========================================
# 资源清理
# ==========================================
def destroy_rag_system(rag_chain):
    try:
        if hasattr(rag_chain, '_model'):
            del rag_chain._model
        if hasattr(rag_chain, '_tokenizer'):
            del rag_chain._tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"清理失败: {e}")


# ==========================================
# Streamlit UI
# ==========================================
st.set_page_config(page_title=ST_TITLE, page_icon="🏥")
st.title(ST_TITLE)
st.markdown("### 💊 基于医学知识库的智能问答系统（动态 MMR + 上下文感知 BGE-Reranker）")
st.markdown("---")

with st.sidebar:
    st.header("🔍 优化特性")
    st.info("• 动态 MMR λ（高风险问题更相关）")
    st.info("• BGE-Reranker 上下文感知 + 无答案检测")
    st.info("• Token-aware 对话历史截断")
    st.info("• 结构化 Prompt + 安全兜底")

    if "rag_chain" not in st.session_state:
        with st.spinner("正在加载医学知识库与模型..."):
            st.session_state.rag_chain, msg = initialize_rag_system()

    if st.session_state.rag_chain:
        st.success("✅ RAG 系统已就绪")
        st.info(f"🧠 LLM: {os.path.basename(MODEL_NAME)}")
        st.info(f"reranker: BGE-Reranker-v2-m3 (阈值={SCORE_THRESHOLD})")
    else:
        st.error(f"❌ 初始化失败: {msg}")
        st.stop()

    st.markdown("---")
    st.markdown("**免责声明**")
    st.markdown("⚠️ 本系统仅提供医学知识参考，不能替代专业医疗建议。")

    if st.button("清除对话历史"):
        st.session_state.messages = []
        st.rerun()

# 初始化对话历史（并净化）
if "messages" not in st.session_state:
    st.session_state.messages = []
else:
    # 可选：启动时净化旧会话
    st.session_state.messages = normalize_messages(st.session_state.messages)

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 用户输入处理
if prompt := st.chat_input("请输入关于中文医疗领域的问题..."):
    # 显示并保存用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # ✅ 关键：传入包含 messages 的字典
    input_dict = {
        "question": prompt,
        "messages": st.session_state.messages
    }

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        try:
            for chunk in st.session_state.rag_chain.stream(input_dict):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)
        except Exception as e:
            error_msg = f"系统错误: {str(e)}"
            st.error(error_msg)
            full_response = error_msg

    # 保存助手回复
    st.session_state.messages.append({"role": "assistant", "content": full_response})