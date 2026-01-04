import streamlit as st
import torch
import os
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
import chromadb

# ================= 配置区域 =================
# 页面配置
st.set_page_config(page_title="医疗 RAG 助手", page_icon="🏥", layout="wide")

# 本地路径
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "medical_rag"
EMBED_MODEL_PATH = "./BAAI/bge-m3"
LLM_MODEL_PATH = "./Qwen/Qwen-Medical-Merged"

# ================= 模型加载=================
@st.cache_resource
def load_rag_engine():
    status = st.empty()
    status.info("🚀 正在初始化 RAG 系统 (加载模型中，请稍候)...")

    # 1. 加载 Embedding 模型
    embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_PATH,
        device="cuda",
        trust_remote_code=True
    )
    Settings.embedding_model = embed_model

    # 2. 加载 LLM (Qwen-2.5)
    llm = HuggingFaceLLM(
        context_window=32000, 
        max_new_tokens=512,
        generate_kwargs={"temperature": 0.1, "do_sample": True},
        tokenizer_name=LLM_MODEL_PATH,
        model_name=LLM_MODEL_PATH,
        device_map="auto",
        model_kwargs={"torch_dtype": torch.float16, "load_in_4bit": True},
    )
    Settings.llm = llm

    # 3. 连接向量数据库
    if not os.path.exists(CHROMA_PATH):
        st.error("❌ 未找到向量库！请先运行构建脚本。")
        st.stop()
        
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # 4. 加载索引
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model
    )

    # 5. 构建聊天引擎
    # mode="condense_plus_context": 适合多轮对话，会把历史记录压缩成新的查询
    chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        system_prompt="""你是一个专业的医疗智能助手。
        请严格根据提供的【参考文档】回答用户的医疗问题。
        如果文档中没有答案，请直接说明“资料库中未找到相关信息”，不要编造。
        回答时请保持客观、严谨，并使用中文。""",
        similarity_top_k=3,
        verbose=True
    )
    
    status.empty() # 清除加载提示
    return chat_engine

# ================= 界面逻辑 =================

st.title("🏥 华驼医疗 RAG 问答系统")
st.caption("基于 Qwen-2.5-7B 与 Huatuo-26M 构建 | 支持多轮对话与引用溯源")

# 初始化 Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "engine" not in st.session_state:
    st.session_state.engine = load_rag_engine()

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # 如果历史消息里有引用源，也显示出来
        if "sources" in message:
            with st.expander("📚 参考来源 (历史)"):
                st.markdown(message["sources"])

# 处理用户输入
if prompt := st.chat_input("请输入症状或医疗问题..."):
    # 1. 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 生成回答
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("🔍 正在检索文献并生成回答..."):
            try:
                # 调用 RAG 引擎
                response = st.session_state.engine.chat(prompt)
                full_response = response.response
                
                # 展示回答
                message_placeholder.markdown(full_response)
                
                # --- 关键：解析并展示引用来源 ---
                source_text = ""
                if response.source_nodes:
                    with st.expander("📚 查看参考来源 (Evidence)"):
                        for idx, node in enumerate(response.source_nodes):
                            # 获取元数据
                            meta = node.metadata
                            score = node.score
                            content = node.node.get_content()[:100] + "..." # 只显示前100字预览
                            
                            # 格式化显示
                            one_source = f"**[来源 {idx+1}]** (相似度: {score:.2f})\n\n> {content}\n"
                            st.markdown(one_source)
                            st.divider()
                            source_text += one_source

                # 保存助手回复到历史记录
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "sources": source_text # 记录引用以便后续查看
                })
                
            except Exception as e:
                st.error(f"发生错误: {str(e)}")