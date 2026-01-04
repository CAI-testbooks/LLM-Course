import streamlit as st
import torch
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from threading import Thread

# ==========================================
# 1. 配置区域
# ==========================================
ST_TITLE = "🤖 ControlExpert 2.0 - 自动化专业智能终端"
# 请确保路径正确
BASE_MODEL_PATH = r"D:\workerspace\models\Qwen\Qwen2___5-1___5B-Instruct"
DB_PATH = r"D:\workerspace\control_qa\vector_db"


# ==========================================
# 2. 核心逻辑：资源加载 (单例模式)
# ==========================================
@st.cache_resource
def load_control_expert_core():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",
        dtype=torch.float16,
        trust_remote_code=True
    )
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={'device': 'cuda'}
    )
    vector_db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    return tokenizer, model, vector_db


# ==========================================
# 3. 页面配置
# ==========================================
st.set_page_config(page_title=ST_TITLE, page_icon="🤖", layout="wide")

# 侧边栏
with st.sidebar:
    st.header("🖥️ 系统状态面板")
    with st.spinner("正在唤醒计算核心..."):
        tokenizer, model, vector_db = load_control_expert_core()

    st.success("✅ 系统已就绪")
    st.info(f"📚 知识库: 5050 QA + 教材原文")
    st.info(f"🧠 模型: Qwen2.5-1.5B-Instruct")

    st.markdown("---")

    # [保留布局] 将"清除记录"按钮上移
    if st.button("🔄 清除所有对话记录", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # [保留布局] 利用空行制造“视觉下沉”效果，把设置挤到最下面
    st.markdown("<br>" * 15, unsafe_allow_html=True)

    st.markdown("---")
    # [保留布局] 开发者选项位于最底部
    with st.expander("⚙️ 开发者选项 (高级设置)", expanded=False):
        st.caption("调整 RAG 检索灵敏度与调试模式")
        threshold = st.slider("检索相关度阈值", 0.0, 1.0, 0.35, 0.05)
        debug_mode = st.toggle("🛠️ 开启调试模式", value=False)

st.title(ST_TITLE)
st.caption("专注自动控制原理专业问答。")


# ==========================================
# 🔥 核心修复：智能感知型 LaTeX 渲染引擎 (增强版)
# ==========================================
def format_latex(text):
    """
    ControlExpert 2.0 渲染引擎 - 智能感知版
    """
    # 1. 清理 Markdown 代码块干扰
    # 【关键修改】不再简单删除反引号，而是将它们转换为 LaTeX 的 $ 符号
    # 这样模型输出的 `s^2` (代码块) 会变成 $s^2$ (数学公式)
    text = text.replace("```latex", "").replace("```", "")  # 先移除大块代码标记
    text = text.replace("`", "$")  # 将行内代码标记转换为行内公式标记

    # 2. 统一 LaTeX 括号标准
    text = text.replace(r"\[", "\n$$\n")
    text = text.replace(r"\]", "\n$$\n")
    text = text.replace(r"\(", "$")
    text = text.replace(r"\)", "$")

    # 3. 换行符增强
    text = text.replace(r"\begin", "@@BEGIN@@")
    text = text.replace(r"\end", "@@END@@")
    text = text.replace(r"\hline", "@@HLINE@@")
    text = text.replace(r"\frac", "@@FRAC@@")

    text = re.sub(r"\\{2,}", r"\\\\\\\\", text)
    text = re.sub(r"([^\\])\\\s+s", r"\1\\\\\\\\ s", text)

    text = text.replace("@@BEGIN@@", r"\begin")
    text = text.replace("@@END@@", r"\end")
    text = text.replace("@@HLINE@@", r"\hline")
    text = text.replace("@@FRAC@@", r"\frac")

    # 4. 【智能封装】 Smart Wrapping
    def smart_wrap_start(match):
        prefix = match.group(1) or ""
        content = match.group(2)
        if "$$" in prefix:
            return match.group(0)
        else:
            return f"\n$$\n{content}"

    text = re.sub(r"(\$\$\s*)?(\\begin\{.*?\})", smart_wrap_start, text, flags=re.IGNORECASE)

    def smart_wrap_end(match):
        content = match.group(1)
        suffix = match.group(2) or ""
        if "$$" in suffix:
            return match.group(0)
        else:
            return f"{content}\n$$\n"

    text = re.sub(r"(\\end\{.*?\})(\s*\$\$)?", smart_wrap_end, text, flags=re.IGNORECASE)

    # 5. 最终清洗
    text = re.sub(r"\$\$\s*\$\$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


# ==========================================
# 4. 对话逻辑
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 输入框
if prompt := st.chat_input("请输入自控原理问题..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("正在检索教材库..."):
        # 获取 Top-3 结果
        results = vector_db.similarity_search_with_relevance_scores(prompt, k=3)
        # 根据阈值过滤
        valid_docs = [doc for doc, score in results if score > threshold]

    with st.chat_message("assistant"):
        # 逻辑优化：如果没有检索到内容，给用户更明确的提示
        if not valid_docs:
            response = f"⚠️ **检索未命中**\n\n当前相关度阈值设为 `{threshold}`，未在知识库中找到足够相关的知识点。\n\n💡 **建议**：\n1. 尝试在左侧“开发者选项”中降低阈值 (例如调至 0.3)。\n2. 换一种更准确的提问方式。"
            st.warning(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            context = "\n".join([doc.page_content for doc in valid_docs])

            # ================= [ 页码提取逻辑 ] =================
            page_numbers = []
            for doc in valid_docs:
                # 获取 page 字段，如果没有则默认为 "未知"
                # 强制转字符串，防止 int/str 混用导致报错
                p = doc.metadata.get("page", "未知")
                page_numbers.append(str(p))

            # 去重并排序 (智能排序：优先按数字大小，否则按字典序)
            unique_pages = sorted(list(set(page_numbers)), key=lambda x: int(x) if x.isdigit() else 0)
            pages_str = ", ".join(unique_pages)

            # 构造新的来源字符串
            sources = f"教材原文 - 第 {pages_str} 页"
            # ===================================================

            # 🚀 提示词增强
            system_prompt = r"""你是一个自动控制原理专家。
1. 所有数学公式必须使用 LaTeX 格式。
2. 【劳斯表/矩阵专用规则】
   - 必须使用 `\begin{array}` 环境构建。
   - **严禁**使用 Markdown 表格 (|---|)。
   - **严禁**使用 Markdown 代码块 (```)。
   - 每一行结束时，请严格输出 `\\` (双反斜杠) 表示换行。
   - **绝对不要**输出 `\\\` (三斜杠) 或 `\ \` (单斜杠空格)。
3. 遇到矩阵或复杂算式，请直接输出公式块，不要加多余的解释文本。
4. **正确示例：**
   $$
   \begin{array}{c|cc}
   s^2 & 1 & 2 \\
   s^1 & 3 & 4 \\
   s^0 & 5 & 0
   \end{array}
   $$"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"参考资料：\n{context}\n\n问题：{prompt}"}
            ]

            input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                      return_tensors="pt").to(model.device)
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            gen_kwargs = dict(input_ids=input_ids, streamer=streamer, max_new_tokens=1024, temperature=0.3)
            Thread(target=model.generate, kwargs=gen_kwargs).start()

            response_placeholder = st.empty()
            full_response = ""
            for new_text in streamer:
                full_response += new_text
                # 实时应用清洗逻辑
                display_text = format_latex(full_response)
                response_placeholder.markdown(display_text + "▌")

            # 最终显示
            final_formatted = format_latex(full_response)
            final_display = final_formatted + f"\n\n--- \n 📚 **参考来源**: {sources}"
            response_placeholder.markdown(final_display)

            # 🔥 调试信息展示区 (已修复报错误报)
            if debug_mode:
                with st.expander("🛠️ 工程师调试视图 (Raw Data)", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption("1. 模型原始输出 (Raw)")
                        st.code(full_response, language="latex")
                    with col2:
                        st.caption("2. 清洗后数据 (Formatted)")
                        st.code(final_formatted, language="latex")

                    # 【修复】逻辑优化：只要有 $ 或 $$ 都算正常，不再误报红色警告
                    if "$$" in final_formatted or "$" in final_formatted:
                        st.success("✅ 检测到 LaTeX 数学符号，渲染引擎正常工作中。")
                    else:
                        st.info("ℹ️ 当前输出为纯文本（无数学公式），这是正常的。")

            st.session_state.messages.append({"role": "assistant", "content": final_display})