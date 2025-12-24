import streamlit as st
import os
import time

# 引入 LangChain 组件
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ==========================================
# 配置区域
# ==========================================
ST_TITLE = "🚁 X-2000 无人机 - 智能技术支持终端"
os.environ["OPENAI_API_KEY"] = "sk-gyuofotkkugmqvlmcuchjdzmipktruzczqvqtqyiyfqbqvsu"  # 填入你的 Key
os.environ["OPENAI_API_BASE"] = "https://api.siliconflow.cn/v1"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
EMBEDDING_MODEL = "BAAI/bge-m3"


# ==========================================
# 核心逻辑 (使用 @st.cache_resource 缓存，防止每次刷新都重跑)
# 技术点：Singleton 模式在 Web 开发中的应用
# ==========================================
@st.cache_resource
def initialize_rag_system():
    """
    初始化 RAG 系统：加载数据 -> 切分 -> 向量化 -> 存储
    只运行一次，后续直接调用缓存的对象。
    """
    # 1. 加载数据
    script_dir = os.path.dirname(os.path.abspath(__file__))
    knowledge_path = os.path.join(script_dir, "knowledge.txt")
    if not os.path.exists(knowledge_path):
        return None, f"找不到 knowledge.txt 文件: {knowledge_path}"

    loader = TextLoader(knowledge_path, encoding="utf-8")
    docs = loader.load()

    # 2. 切分
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    splits = splitter.split_documents(docs)

    # 3. 向量化
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    # 4. 构建检索器
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 5. 定义 LLM (开启流式输出 streaming=True)
    llm = ChatOpenAI(
        model_name=MODEL_NAME,
        temperature=0.1,
        streaming=True
    )

    # 6. 定义 Prompt
    template = """
    你是一个专业的无人机技术支持专家。请结合以下【上下文】和【历史聊天记录】回答用户问题。
    如果不知道，请直接说不知道。

    【上下文】：
    {context}

    【用户问题】：
    {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 7. 构建链
    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return rag_chain, "系统初始化完成"


# ==========================================
# Streamlit UI 界面逻辑
# ==========================================
st.set_page_config(page_title=ST_TITLE, page_icon="🚁")
st.title(ST_TITLE)

# 侧边栏：显示系统状态
with st.sidebar:
    st.header("系统状态面板")
    with st.spinner("正在启动神经中枢..."):
        rag_chain, msg = initialize_rag_system()

    if rag_chain:
        st.success("✅ 知识库已挂载 (RAG Ready)")
        st.info(f"🧠 模型: {MODEL_NAME}")
    else:
        st.error(f"❌ 启动失败: {msg}")
        st.stop()

    st.markdown("---")
    if st.button("清除对话历史"):
        st.session_state.messages = []
        st.rerun()

# 初始化对话历史 (Session State)
# 技术点：Web 是无状态的，必须手动维护上下文
if "messages" not in st.session_state:
    st.session_state.messages = []

# 1. 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. 处理用户输入
if prompt := st.chat_input("请输入关于 X-2000 的问题..."):
    # 显示用户的问题
    st.chat_message("user").markdown(prompt)
    # 将问题存入历史
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 3. 生成回答 (流式)
    with st.chat_message("assistant"):
        response_placeholder = st.empty()  # 占位符
        full_response = ""

        # 调用 RAG 链 (Stream 模式)
        try:
            # 这是一个生成器，会不断吐出字符
            for chunk in rag_chain.stream(prompt):
                full_response += chunk
                # 实时刷新界面，模拟打字机效果
                response_placeholder.markdown(full_response + "▌")
                # time.sleep(0.01) # 如果由于网络太快看不清流式，可以取消注释

            response_placeholder.markdown(full_response)
        except Exception as e:
            st.error(f"发生错误: {e}")
            full_response = f"抱歉，系统遇到故障: {e}"

    # 将 AI 的回答存入历史
    st.session_state.messages.append({"role": "assistant", "content": full_response})