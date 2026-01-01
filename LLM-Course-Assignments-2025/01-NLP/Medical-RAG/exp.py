import streamlit as st
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"#从 Hugging Face 下载 BAAI/bge-m3 嵌入模型时 无法连接到互联网  需修改
import json
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
# ==========================================
# 配置区域
# ==========================================

ST_TITLE = "中文医疗领域智能问答系统"
MODEL_NAME = "/root/autodl-tmp/qwen/Qwen2___5-7B-Instruct"  # 本地模型路径
EMBEDDING_MODEL = "BAAI/bge-m3"
VECTOR_DB_PATH = "./chroma_db_medical"  # ← 向量库持久化目录 本地已存在 Chroma 向量数据库（如 ./chroma_db_history），就直接加载；如果不存在，则从文档构建并向磁盘保存。
# ==========================================
# 自定义 JSONL 加载函数
# ==========================================
def load_jsonl_as_documents(file_path):
    """从 JSONL 文件加载为 LangChain Documents"""
    docs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                questions = data.get("questions", [])
                answers = data.get("answers", [])
                
                # 支持多个问题对应一个答案（取第一个问题作为 content）
                if not questions or not answers:
                    st.warning(f"跳过无效行 {file_path}:{line_num}")
                    continue
                
                # 取第一个问题（可能是列表嵌套）
                question = questions[0]
                if isinstance(question, list):
                    question = question[0] if question else ""
                
                answer = answers[0] if answers else ""
                
                # 构造文本内容（可选：只用问题，或问题+答案）
                text = f"问题：{question}\n答案：{answer}"
                metadata = {
                    "question": question,
                    "answer": answer,
                    "source": os.path.basename(file_path),
                    "line": line_num
                }
                docs.append(Document(page_content=text, metadata=metadata))
            except json.JSONDecodeError as e:
                st.error(f"JSON 解析失败 {file_path}:{line_num} - {e}")
                continue
    return docs

# ==========================================
# 初始化 RAG 系统
# ==========================================
@st.cache_resource
def initialize_rag_system():
    dataset_dir = "/root/autodl-tmp/Medical-RAG/dataset"
    if not os.path.exists(dataset_dir):
        return None, f"找不到数据集目录: {dataset_dir}"

    # 向量化配置（必须提前定义，用于加载或创建）
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # 检查是否已有持久化的向量库
    if os.path.exists(VECTOR_DB_PATH):
        st.info("检测到已有向量库，正在加载...")
        vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
        st.success("✅ 向量库加载完成！")
    else:
        # === 需要重新构建向量库 ===
        json_files = [
            os.path.join(dataset_dir, "test_data.json"),
            os.path.join(dataset_dir, "validation_data.json"),
            os.path.join(dataset_dir, "train_data_8k.json")
        ]
        
        docs = []
        for file_path in json_files:
            if os.path.exists(file_path):
                st.info(f"正在加载: {os.path.basename(file_path)}")
                file_docs = load_jsonl_as_documents(file_path)
                docs.extend(file_docs)
                st.success(f"完成加载: {len(file_docs)} 条记录 from {os.path.basename(file_path)}")
            else:
                st.warning(f"文件不存在: {file_path}")

        if not docs:
            return None, "未加载到任何有效文档"

        # 切分
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        splits = splitter.split_documents(docs)

        st.info("正在构建向量库")#首次加载巨慢无比 耐心等待
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=VECTOR_DB_PATH
        )
        st.success("✅ 向量库构建完成并已保存至本地！")

    # 检索器
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 加载本地 Qwen-2.5-7B 模型（带量化以节省显存）
    # 加载 tokenizer 并修复 pad token
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # ← 关键修复！

    bnb_config = BitsAndBytesConfig(
        #load_in_4bit=True,#必须注释，否则会出现梳理值naf 等问题
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # 创建 pipeline 时显式指定 pad/eos token
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        temperature=0.1,
        top_p=0.95,
        repetition_penalty=1.1,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,   # ← 必须
        eos_token_id=tokenizer.eos_token_id,   # ← 推荐
        clean_up_tokenization_spaces=True
    )
    #llm_pipeline = HuggingFacePipeline(pipeline=pipe)

    #llm = HuggingFacePipeline(pipeline=pipe)
    
    # 正确方式：先包装成 HuggingFacePipeline，再用 ChatHuggingFace
    llm_pipeline = HuggingFacePipeline(pipeline=pipe)

    llm = ChatHuggingFace(
        llm=llm_pipeline,       # ← 必须用 llm= 参数
        tokenizer=tokenizer,
        streaming=True
    )
    # Prompt
    template = """
    你是一个专业的医疗AI助手。请结合以下【医学知识】回答用户问题。
    如果不知道，请直接说"根据现有医学资料，我无法提供确切答案，建议咨询专业医生"。

    【医学知识】：
    {context}

    【用户问题】：
    {question}
    """
    prompt = ChatPromptTemplate.from_template(template)


    # RAG 链
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, "系统初始化完成"


# ==========================================
# Streamlit UI
# ==========================================
st.set_page_config(page_title=ST_TITLE, page_icon="🏥")
st.title(ST_TITLE)
st.markdown("### 💊 基于医学知识库的智能问答系统")
st.markdown("---")

with st.sidebar:
    st.header("系统状态面板")
    with st.spinner("正在加载医学知识库..."):
        rag_chain, msg = initialize_rag_system()

    if rag_chain:
        st.success("✅ 医学知识库已挂载 (RAG Ready)")
        st.info(f"🧠 模型: {MODEL_NAME}")
    else:
        st.error(f"❌ 启动失败: {msg}")
        st.stop()

    st.markdown("---")
    st.markdown("**免责声明**")
    st.markdown("⚠️ 本系统仅提供医学知识参考，不能替代专业医疗建议。如有紧急情况，请立即就医。")
    
    if st.button("清除对话历史"):
        st.session_state.messages = []
        st.rerun()

# 初始化对话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 用户输入
if prompt := st.chat_input("请输入关于中文医疗领域的问题..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        try:
            for chunk in rag_chain.stream(prompt):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)
        except Exception as e:
            error_msg = f"抱歉，系统遇到错误: {str(e)}"
            st.error(error_msg)
            full_response = error_msg

    st.session_state.messages.append({"role": "assistant", "content": full_response})