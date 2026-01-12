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

ST_TITLE = "通用中文医疗领域智能问答系统"
#MODEL_NAME = "/root/autodl-tmp/qwen/Qwen2___5-7B-Instruct"  # 本地模型路径
MODEL_NAME = "/root/autodl-tmp/Medical-RAG/Tune-model/medical-qwen-merged"  # 修改为merage后的模型路径
EMBEDDING_MODEL = "BAAI/bge-m3"
VECTOR_DB_PATH = "/root/autodl-tmp/Medical-RAG/chroma_db_medical"  # ← 向量库持久化目录 本地已存在 Chroma 向量数据库（如 ./chroma_db_medical），就直接加载；如果不存在，则从文档构建并向磁盘保存。
# ==========================================
# 自定义 JSONL 加载函数
# ==========================================
def load_alpaca_json_as_documents(file_path):
    """
    从 Alpaca 格式的 JSON 数组文件加载为 LangChain Documents
    适配格式：[{"instruction": "问题", "input": "", "output": "答案"}, ...]
    
    Args:
        file_path (str): JSON文件路径（Alpaca格式，数组）
        拼接问题+答案
    
    Returns:
        list[Document]: LangChain Document列表
    """
    docs = []
    try:
        # ✅ 关键修改1：整文件加载JSON数组（替代JSONL逐行读取）
        with open(file_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)  # 直接加载整个JSON数组
        
        # 遍历每个Alpaca格式的条目
        for idx, item in enumerate(data_list, 1):
            # ✅ 关键修改2：字段映射（Alpaca格式→QA）
            # 提取问题（instruction）、答案（output），input为空则忽略
            instruction = item.get("instruction", "").strip()  # 对应原问题
            output = item.get("output", "").strip()            # 对应原答案
            input_text = item.get("input", "").strip()         # Alpaca的input字段（医疗场景为空）

            # 过滤无效条目（无问题/无答案）
            if not instruction or not output:
                st.warning(f"跳过无效条目（索引{idx}）：无问题或无答案")
                continue

            # ✅ 现在总是拼接问题+答案（不再有条件判断）
            page_content = f"问题：{instruction}\n答案：{output}"

            # ✅ 元数据优化：保留原始信息用于追踪
            metadata = {
                "source_instruction": instruction,  # 原始问题（替代原source_question）
                "source_file": os.path.basename(file_path),
                "item_index": idx,  # 条目索引（替代原行号，JSON数组无行号）
                "has_input": True if input_text else False  # 标记是否有input（医疗场景为False）
            }

            # 创建LangChain Document对象并加入列表
            docs.append(Document(page_content=page_content, metadata=metadata))

    except json.JSONDecodeError as e:
        st.error(f"JSON解析失败：{file_path} 不是合法的JSON数组格式 - {e}")
        return []
    except FileNotFoundError:
        st.error(f"文件不存在：{file_path}")
        return []
    except Exception as e:
        st.error(f"加载文件出错：{e}")
        return []

    st.success(f"成功加载 {len(docs)} 条有效医疗QA数据（来自{os.path.basename(file_path)}）")
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
            os.path.join(dataset_dir, "alpaca_formatted_test_data.json"),
            os.path.join(dataset_dir, "alpaca_formatted_validation_data.json"),
            os.path.join(dataset_dir, "alpaca_formatted_train_data.json") # 如果是构建RAG系统的话，使用全部的answer进行向量数据库持久化即可，后续加入QA对进行微调即可
        ]
        
        docs = []
        for file_path in json_files:
            if os.path.exists(file_path):
                st.info(f"正在加载: {os.path.basename(file_path)}")
                file_docs = load_alpaca_json_as_documents(file_path)# ← 拼接Q A对
                docs.extend(file_docs)
                st.success(f"完成加载: {len(file_docs)} 条记录 from {os.path.basename(file_path)}")
            else:
                st.warning(f"文件不存在: {file_path}")

        if not docs:
            return None, "未加载到任何有效文档"

       
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        splits = splitter.split_documents(docs)

        st.info("正在构建向量库")
        #首次加载巨慢无比 耐心等待 大约几分钟
        #首次开启页面卡顿后，可以重新python -m streamlit run exp.py ，再打开页面巨快
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=VECTOR_DB_PATH
        )
        st.success("✅ 向量库构建完成并已保存至本地！")

    # 检索器 - 调整k值以适应更大的chunk_size
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 加载本地 Qwen-2.5-7B 模型（带量化以节省显存）
    # 加载 tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    # 加载 model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,    
        device_map="auto",
        trust_remote_code=True,
        #load_in_4bit=True,#显存不够可以打开

    )

    # 创建 pipeline 时显式指定 pad/eos token
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.95,
        repetition_penalty=1.1,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,   # ← 必须
        eos_token_id=tokenizer.eos_token_id,   # ← 推荐
        clean_up_tokenization_spaces=True,
        early_stopping=True      # 生成到结束符自动停止，避免冗余
    )
    #llm_pipeline = HuggingFacePipeline(pipeline=pipe)

    #llm = HuggingFacePipeline(pipeline=pipe)
    
    
    llm_pipeline = HuggingFacePipeline(pipeline=pipe)

    llm = ChatHuggingFace(
        llm=llm_pipeline,       # ← 必须用 llm= 参数
        tokenizer=tokenizer,
        streaming=True
    )

    # 同时优化Prompt模板（减少模型无意义列举）
    template = """
    你是一个专业的医学助手。
    回答要求：1. 条理清晰；2. 禁止重复表述；3. 生成答案时，不做冗余推理
    如果不知道，请直接说"根据现有医学资料，我无法提供确切答案，建议咨询专业医生"。

    医学知识：
    {context}

    用户问题：
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