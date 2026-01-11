import streamlit as st
from model.llm_api import DeepSeekAPI

# 初始化会话状态
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 初始化大模型API客户端
if "llm_client" not in st.session_state:
    st.session_state.llm_client = DeepSeekAPI()

# 页面标题
st.title("医学智能问答机器人")
st.divider()

# 用户输入框
user_input = st.chat_input("请输入你的医学问题：")

if user_input:
    # 输入添加到历史记录
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # 显示所有历史消息
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 调用后端API获取回答
    with st.spinner("AI正在分析问题..."):
        assistant_reply = st.session_state.llm_client.get_response(st.session_state.chat_history)

        # 回答添加到历史记录并显示
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})
        with st.chat_message("assistant"):
            st.markdown(assistant_reply)

# streamlit run D:\PYTHON\PycharmProjects\large_model_rag\src\chat_web.py
