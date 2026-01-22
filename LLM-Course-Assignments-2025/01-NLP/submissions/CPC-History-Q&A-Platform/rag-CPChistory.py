import streamlit as st
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langchain_community.document_loaders import TextLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
from tqdm import tqdm
import glob
import numpy as np
# ==========================================
# 配置区域（针对党史数据优化）
# ==========================================
ST_TITLE = "党史知识 RAG 问答系统"

# 数据路径配置（修改为你的党史书籍路径）
TXT_FOLDER = "./data"  # 存放党史相关txt文件
SUPPORTED_EXTENSIONS = ["*.txt"]  # 支持多种格式
# 本地 Qwen3-0.6-Instruct 路径
LOCAL_MODEL_PATH = "/root/model/Qwen/Qwen3-0___6B"
# 嵌入模型优化为更适合中文党史内容
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"  # 对中文支持更好[6]
# 向量库持久化目录
VECTOR_DB_PATH = "./chroma_db_party_history"

# 文本分割参数优化（针对党史文献特点）
CHUNK_SIZE = 600  # 党史文献通常结构清晰，适当减小chunk大小
CHUNK_OVERLAP = 80
RETRIEVE_COUNT = 5  # 检索数量调整


# ==========================================
# 初始化 RAG 系统（优化版本）
# ==========================================
@st.cache_resource
def initialize_rag_system():
    """
    初始化党史RAG系统，针对党史文献特点进行优化
    """
    # 1. 检查数据文件夹并查找支持的文件
    if not os.path.exists(TXT_FOLDER):
        return None, f"数据文件夹不存在: {TXT_FOLDER}"

    # 查找所有支持的文件格式
    data_files = []
    for extension in SUPPORTED_EXTENSIONS:
        data_files.extend(glob.glob(os.path.join(TXT_FOLDER, extension)))

    if not data_files:
        return None, f"文件夹 {TXT_FOLDER} 中没有找到支持的文档文件(txt/md/pdf)"

    st.info(f"发现 {len(data_files)} 个党史文档文件，正在加载...")

    # 2. 加载文档（优化错误处理）
    docs = []
    failed_files = []

    for file_path in tqdm(data_files, desc="加载党史文档"):
        try:
            if file_path.endswith('.pdf'):
                # 对于PDF文件使用更强大的加载器
                loader = UnstructuredFileLoader(file_path, strategy="fast")
            else:
                loader = TextLoader(file_path, encoding="utf-8")

            file_docs = loader.load()
            # 为每个文档添加元数据，记录来源文件
            for doc in file_docs:
                doc.metadata["source"] = os.path.basename(file_path)
            docs.extend(file_docs)
        except Exception as e:
            failed_files.append((os.path.basename(file_path), str(e)))
            continue

    if failed_files:
        st.warning(f"部分文件加载失败: {[f[0] for f in failed_files]}")

    if not docs:
        return None, "所有文件加载失败，请检查文件格式和编码"

    st.success(f"成功加载 {len(docs)} 个文档片段")

    # 3. 文本切分优化（针对党史文献特点）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n## ", "\n# ", "\n\n", "\n", "。", "！", "？", "；"],  # 中文友好分隔符
    )
    splits = text_splitter.split_documents(docs)
    st.info(f"切分为 {len(splits)} 个文本块")

    # 4. 嵌入模型优化
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={
            'normalize_embeddings': True,
            'batch_size': 32  # 优化批处理大小
        }
    )

    # 5. 构建或加载向量库（添加集合名称避免冲突）
    if os.path.exists(VECTOR_DB_PATH):
        st.info("检测到已有向量库，直接加载...")
        vectorstore = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=embeddings,
            collection_name="party_history_collection"
        )
    else:
        st.info("正在构建党史知识向量库（首次运行较慢）...")
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=VECTOR_DB_PATH,
            collection_name="party_history_collection"
        )
        st.success("党史知识向量库构建完成并已保存！")

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVE_COUNT}
    )

    # 6. 加载模型
    if not os.path.exists(LOCAL_MODEL_PATH):
        return None, f"模型路径不存在: {LOCAL_MODEL_PATH}"

    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # load_in_4bit=True,   # 如显存不够可开启（需 pip install bitsandbytes）
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        temperature=0.3,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1
    )

    # 正确方式：先包装成 HuggingFacePipeline，再用 ChatHuggingFace
    llm_pipeline = HuggingFacePipeline(pipeline=pipe)

    llm = ChatHuggingFace(
        llm=llm_pipeline,  # ← 必须用 llm= 参数
        tokenizer=tokenizer,
        streaming=True
    )

    # 7. Prompt模板优化（针对党史问答特点）
    template = """
你是一个党史研究专家，请根据以下检索到的上下文，准确、详尽地回答用户关于党史的问题。

党史知识具有严肃性和准确性要求，请确保：
1. 回答要基于事实，准确引用历史事件的时间、地点和人物
2. 对于重要历史事件和决策，要体现其历史背景和意义
3. 如果上下文信息不足，请明确说明并建议查阅权威党史资料
4. 回答要体现党史教育的严肃性和教育意义

上下文：
{context}

问题：{question}

请根据以上上下文提供专业、准确的党史知识回答：
"""
    prompt = ChatPromptTemplate.from_template(template)

    # 8. RAG Chain
    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return rag_chain, f"系统就绪！知识库包含 {len(data_files)} 个党史文档"


# ==========================================
# Streamlit 界面（优化用户体验）
# ==========================================
st.set_page_config(
    page_title=ST_TITLE,
    page_icon="🇨🇳",  # 改为更符合党史主题的图标
    layout="wide"
)

st.title(ST_TITLE)
st.markdown("### 基于本地大模型的党史知识智能问答系统")

with st.sidebar:
    st.header("📊 系统状态")
    with st.spinner("正在初始化党史RAG系统..."):
        rag_chain, msg = initialize_rag_system()

    if rag_chain:
        st.success("✅ RAG 系统已就绪")
        st.info(msg)
        st.info(f"🧠 模型: 本地 Qwen3-0.6B\n\n📚 嵌入模型: {EMBEDDING_MODEL}")

        # 添加使用提示
        st.markdown("---")
        st.header("💡 使用提示")
        st.info("""
        您可以询问关于党史的以下内容：
        - 重要历史事件
        - 党的历次代表大会
        - 重要历史人物
        - 党的理论发展
        - 历史经验和教训
        """)
    else:
        st.error(f"❌ 初始化失败: {msg}")
        st.stop()

    if st.button("🗑️ 清除对话历史"):
        st.session_state.messages = []
        st.rerun()

# 初始化对话历史
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "您好！我是党史知识问答助手，可以为您解答关于中国共产党历史的各种问题。"}
    ]

# 显示历史对话
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 用户输入区域
if prompt := st.chat_input("请输入您想了解的党史相关问题，例如：'中国共产党成立的历史背景是什么？'"):
    # 添加用户消息到历史
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 生成助手回复
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            # 流式输出响应
            for chunk in rag_chain.stream(prompt):
                full_response += chunk
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)

        except Exception as e:
            error_msg = f"生成回答时发生错误: {str(e)}"
            st.error(error_msg)
            full_response = "抱歉，我在处理您的请求时遇到了问题。请稍后再试或尝试重新表述您的问题。"

    # 添加助手回复到历史
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# 添加页脚信息
st.markdown("---")
st.caption("🔍 本系统基于检索增强生成(RAG)技术构建，能够根据提供的党史资料提供准确的问答服务[6](@ref)")