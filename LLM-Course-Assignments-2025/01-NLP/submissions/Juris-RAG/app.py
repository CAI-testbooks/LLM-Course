"""
Juris-RAG Web应用
基于Gradio构建的法律智能问答系统前端
"""
import os
import inspect
from pathlib import Path
from typing import List, Tuple, Generator
from threading import Lock


def _ensure_no_proxy_for_localhost() -> None:
    current = os.getenv("NO_PROXY") or os.getenv("no_proxy") or ""
    hosts = ["127.0.0.1", "localhost"]
    if current:
        parts = [p.strip() for p in current.split(",") if p.strip()]
        lower_parts = {p.lower() for p in parts}
        for host in hosts:
            if host.lower() not in lower_parts:
                parts.append(host)
        value = ",".join(parts)
    else:
        value = ",".join(hosts)
    os.environ["NO_PROXY"] = value
    os.environ["no_proxy"] = value


_ensure_no_proxy_for_localhost()

import gradio as gr
try:
    from gradio.components.textbox import InputHTMLAttributes
except Exception:
    InputHTMLAttributes = None

# 导入配置和RAG引擎
try:
    from src.config import APP_TITLE, APP_DESCRIPTION
    from src.rag_engine import JurisRAGEngine, RAGResponse
except ImportError:
    APP_TITLE = "Juris-RAG 法律智能问答系统"
    APP_DESCRIPTION = "基于RAG技术的中文法律问答系统"
    from rag_engine import JurisRAGEngine, RAGResponse

def _gradio_major_version() -> int:
    try:
        return int(gr.__version__.split(".")[0])
    except Exception:
        return 0


def _supports_kw(callable_obj, name: str) -> bool:
    try:
        return name in inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return False


_CHATBOT_SUPPORTS_TYPE = _supports_kw(gr.Chatbot, "type")
_CHATBOT_EXPECTS_MESSAGES = not _CHATBOT_SUPPORTS_TYPE and _gradio_major_version() >= 6
CHATBOT_FORMAT = "messages" if _CHATBOT_EXPECTS_MESSAGES else "tuples"


def _is_message_item(item) -> bool:
    return isinstance(item, dict) and "role" in item and "content" in item


def _history_is_messages(history) -> bool:
    return isinstance(history, list) and history and all(_is_message_item(m) for m in history)


def _messages_to_tuples(history) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    pending_user = None
    for msg in history:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            pending_user = content
        elif role in ("assistant", "ai", "bot"):
            if pending_user is None:
                continue
            pairs.append((pending_user, content))
            pending_user = None
    return pairs


def _tuples_to_messages(history) -> List[dict]:
    messages: List[dict] = []
    for item in history:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        user, assistant = item
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    return messages


def _normalize_history(history):
    if not history:
        return [], CHATBOT_FORMAT
    if _history_is_messages(history):
        return _messages_to_tuples(history), "messages"
    return list(history), "tuples"


_GRADIO_MAJOR = _gradio_major_version()
_USE_BLOCKS_THEME_CSS = _GRADIO_MAJOR < 6 if _GRADIO_MAJOR else _supports_kw(gr.Blocks, "theme")
_USE_LAUNCH_THEME_CSS = not _USE_BLOCKS_THEME_CSS and (
    _GRADIO_MAJOR >= 6 if _GRADIO_MAJOR else _supports_kw(gr.Blocks.launch, "theme")
)

APP_THEME = gr.themes.Soft(
    primary_hue="cyan",
    secondary_hue="slate",
    neutral_hue="slate",
).set(
    body_background_fill="var(--bg-main)",
    block_background_fill="var(--bg-card)",
    block_border_width="1px",
    block_label_background_fill="var(--bg-panel)",
    input_background_fill="var(--bg-input)",
)


def _load_custom_css() -> str:
    # 优先加载专业版CSS
    css_path = Path(__file__).parent / "assets" / "ui_professional.css"
    if not css_path.exists():
        css_path = Path(__file__).parent / "assets" / "ui.css"
    
    try:
        return css_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


CUSTOM_CSS = _load_custom_css()

# 全局RAG引擎实例
rag_engine = None
_last_submit_guard = {"message": "", "history_key": None}
_last_submit_lock = Lock()


def _is_duplicate_submit(message: str, history) -> bool:
    if not message.strip():
        return False
    history_key = repr(history)
    with _last_submit_lock:
        last_message = _last_submit_guard.get("message", "")
        last_history_key = _last_submit_guard.get("history_key")
        if message == last_message and history_key == last_history_key:
            return True
        _last_submit_guard["message"] = message
        _last_submit_guard["history_key"] = history_key
    return False


def initialize_engine():
    """初始化RAG引擎"""
    global rag_engine
    if rag_engine is None:
        try:
            rag_engine = JurisRAGEngine(streaming=False)
            return True, "RAG引擎初始化成功！"
        except FileNotFoundError as e:
            return False, "向量数据库未找到，请先运行数据处理脚本：\npython -m src.data_processing"
        except ValueError as e:
            return False, f"API配置错误：{str(e)}"
        except Exception as e:
            return False, f"初始化失败：{str(e)}"
    return True, "RAG引擎已就绪"


def format_citations(citations) -> str:
    """格式化引用来源为Markdown"""
    if not citations:
        return "暂无引用来源"
    
    citation_md = ""
    for i, citation in enumerate(citations, 1):
        citation_md += f"**[{i}] {citation.source}**\n"
        citation_md += f"- 类型: {citation.doc_type}\n"
        
        # 添加额外元数据
        if citation.metadata.get("accusation"):
            citation_md += f"- 罪名: {citation.metadata['accusation']}\n"
        if citation.metadata.get("articles"):
            citation_md += f"- 相关法条: 第{citation.metadata['articles']}条\n"
        if citation.metadata.get("article"):
            citation_md += f"- 条款: {citation.metadata['article']}\n"
        
        # 内容预览
        citation_md += f"- 内容摘要: {citation.content[:150]}...\n\n"
    
    return citation_md


def chat_response(
    message: str,
    history
) -> Tuple[str, str, str, List]:
    """
    处理用户消息并返回响应
    
    Args:
        message: 用户输入的消息
        history: 对话历史
        
    Returns:
        tuple: (回答, 引用信息, 置信度信息, 更新后的历史)
    """
    global rag_engine
    
    if not message.strip():
        return "", "请输入问题", "", history
    
    # 确保引擎已初始化
    if rag_engine is None:
        success, msg = initialize_engine()
        if not success:
            return msg, "", "", history
    
    try:
        tuples_history, history_format = _normalize_history(history)
        # 同步历史到引擎，避免共享同一列表被内部追加导致重复
        rag_engine.chat_history = list(tuples_history)
        
        # 获取响应
        response = rag_engine.query(message)
        
        # 格式化引用
        citations_md = format_citations(response.citations)
        
        # 格式化置信度
        if response.confidence >= 0.7:
            confidence_html = f'<div class="confidence-badge confidence-high">高 ({response.confidence:.0%})</div>'
        elif response.confidence >= 0.4:
            confidence_html = f'<div class="confidence-badge confidence-medium">中 ({response.confidence:.0%})</div>'
        else:
            confidence_html = f'<div class="confidence-badge confidence-low">低 ({response.confidence:.0%})</div>'
        
        if response.is_uncertain:
            confidence_html += '<div style="margin-top:8px; font-size:0.85em; color:var(--text-sub)">⚠️ 低置信度回答，仅供参考</div>'
        
        # 更新历史
        new_history_tuples = rag_engine.get_history()
        if history_format == "messages":
            new_history = _tuples_to_messages(new_history_tuples)
        else:
            new_history = new_history_tuples
        
        return response.answer, citations_md, confidence_html, new_history
        
    except Exception as e:
        error_msg = f"处理请求时发生错误: {str(e)}"
        return error_msg, "", "", history


def clear_conversation():
    """清空对话"""
    global rag_engine
    if rag_engine:
        rag_engine.clear_history()
    with _last_submit_lock:
        _last_submit_guard["message"] = ""
        _last_submit_guard["history_key"] = None
    return [], "", "提问后将显示引用来源", "等待提问..."


def search_documents(query: str, top_k: int = 5) -> str:
    """搜索文档并返回Markdown格式"""
    global rag_engine
    
    if not query.strip():
        return "请输入搜索内容"
    
    if rag_engine is None:
        success, msg = initialize_engine()
        if not success:
            return msg
    
    try:
        docs = rag_engine.search_similar(query, k=top_k)
        
        if not docs:
            return "### 📭 未找到相关文档\n请尝试更换关键词。"
        
        result = f"### 🔍 找到 {len(docs)} 个相关文档\n\n"
        
        for i, doc in enumerate(docs, 1):
            source_icon = "📜" if "法" in doc.metadata.get('source', '') else "⚖️"
            result += f"#### {source_icon} 文档 {i}\n"
            result += f"- **来源**: {doc.metadata.get('source', '未知')}\n"
            
            if doc.metadata.get('accusation'):
                result += f"- **罪名**: {doc.metadata['accusation']}\n"
            if doc.metadata.get('article'):
                result += f"- **条款**: {doc.metadata['article']}\n"
            
            # 使用引用块显示内容
            content = doc.page_content.replace('\n', '\n> ')
            result += f"\n> {content}\n\n"
            result += "---\n\n"
        
        return result
        
    except Exception as e:
        return f"搜索时发生错误: {str(e)}"


# 示例问题（更具代表性）
EXAMPLE_QUESTIONS = [
    "故意杀人罪怎么判刑？",
    "盗窃罪的量刑标准是什么？",
    "什么情况下构成正当防卫？",
    "抢劫罪怎么处罚？",
    "未成年人犯罪如何处理？",
    "自首可以减刑吗？"
]


def create_app():
    """创建Gradio应用 (优化版 v2.3)"""
    blocks_kwargs = {"title": APP_TITLE}
    if _USE_BLOCKS_THEME_CSS:
        blocks_kwargs["theme"] = APP_THEME
        blocks_kwargs["css"] = CUSTOM_CSS
    if _supports_kw(gr.Blocks, "fill_height"):
        blocks_kwargs["fill_height"] = True

    with gr.Blocks(**blocks_kwargs) as app:
        # Header
        with gr.Row(elem_id="app-header", elem_classes=["header-container"]):
            with gr.Column(scale=1):
                gr.Markdown("""
                <div class="logo">
                    <span class="logo-icon">⚖️</span> 
                    Juris-RAG
                </div>
                <div class="subtitle">智能法律问答系统 · 刑法专业版</div>
                """)
        
        # Main Container
        with gr.Row(elem_id="app-shell", equal_height=False, elem_classes=["main-layout"]):
            
            # Global Sidebar (Left) - for Actions
            with gr.Column(scale=1, elem_id="sidebar", min_width=240):
                gr.Markdown("### 🛠️ 工具箱", elem_classes=["sidebar-header"])
                new_chat_btn = gr.Button("➕ 新对话", variant="primary", elem_id="new-chat")
                gr.Markdown("---")
                show_sources = gr.Checkbox(value=True, label="显示引用侧栏")
                
                gr.Markdown("#### 💡 使用提示")
                gr.Markdown("- Enter 发送\n- Shift+Enter 换行\n- 支持多轮对话")
                
                init_status = gr.Markdown("🟢 系统已就绪", elem_id="status-indicator")

            # Main Content Area (Tabs)
            with gr.Column(scale=4, elem_id="main-content"):
                with gr.Tabs(elem_id="main-tabs"):
                    
                    # Tab 1: Smart Q&A
                    with gr.Tab("💬 智能问答", elem_id="tab-chat"):
                        with gr.Row():
                            # Chat Area
                            with gr.Column(scale=3, elem_id="chat-col"):
                                with gr.Group(elem_id="chat-group"):
                                    chatbot_kwargs = {
                                        "label": "对话历史",
                                        "show_label": False,
                                        "height": 650,
                                        "elem_id": "chatbot",
                                    }
                                    if _supports_kw(gr.Chatbot, "bubble_full_width"):
                                        chatbot_kwargs["bubble_full_width"] = False
                                    if _CHATBOT_SUPPORTS_TYPE:
                                        chatbot_kwargs["type"] = CHATBOT_FORMAT
                                    if _supports_kw(gr.Chatbot, "show_copy_button"):
                                        chatbot_kwargs["show_copy_button"] = True
                                    if _supports_kw(gr.Chatbot, "avatar_images"):
                                        chatbot_kwargs["avatar_images"] = (None, "⚖️")
                                    
                                    chatbot = gr.Chatbot(**chatbot_kwargs)

                                    with gr.Row(elem_id="composer", elem_classes=["composer-row"]):
                                        msg_input = gr.Textbox(
                                            label="输入您的法律问题",
                                            placeholder="请在此输入具体的法律问题...",
                                            lines=1,
                                            max_lines=6,
                                            scale=8,
                                            show_label=False,
                                            autofocus=True,
                                            elem_id="chat-input"
                                        )
                                        send_btn = gr.Button("发送", variant="primary", scale=1, min_width=80)

                                    gr.Markdown("### 🔍 热门问题")
                                    gr.Examples(
                                        examples=[[q] for q in EXAMPLE_QUESTIONS],
                                        inputs=msg_input,
                                        label="",
                                        elem_id="example-questions"
                                    )

                            # Citation/Info Sidebar (Right) - Collapsible via checkbox
                            with gr.Column(scale=1, elem_id="side-panel", min_width=280, visible=True) as side_panel:
                                with gr.Group():
                                    gr.Markdown("### 📊 置信度", elem_classes=["panel-header"])
                                    confidence_display = gr.Markdown(
                                        value="waiting...",
                                        elem_classes=["panel-section"]
                                    )

                                with gr.Group():
                                    gr.Markdown("### 📚 引用来源", elem_classes=["panel-header"])
                                    citations_display = gr.Markdown(
                                        value="<div style='color:var(--text-light); text-align:center; padding:20px'>提问后将显示法律依据</div>",
                                        elem_classes=["citation-box"]
                                    )

                    # Tab 2: Document Search
                    with gr.Tab("🔎 文档搜索", elem_id="tab-search"):
                        with gr.Group(elem_id="search-group"):
                            gr.Markdown("### 📚 法律法规检索")
                            gr.Markdown("直接检索向量数据库中的法条与案例。")
                            with gr.Row():
                                search_input = gr.Textbox(
                                    label="搜索关键词",
                                    placeholder="例如：故意伤害罪、正当防卫...",
                                    scale=4
                                )
                                search_k = gr.Slider(
                                    minimum=1, maximum=20, value=5, step=1,
                                    label="返回数量", scale=1
                                )
                                search_btn = gr.Button("🔍 搜索", variant="primary", scale=1)

                            search_results = gr.Markdown(
                                value="<br/>请输入关键词开始搜索...",
                                elem_id="search-results"
                            )

                    # Tab 3: System Info
                    with gr.Tab("ℹ️ 系统信息", elem_id="tab-info"):
                        with gr.Group(elem_id="info-group"):
                            gr.Markdown("""
                            # ⚖️ Juris-RAG 法律智能问答系统

                            ## 核心架构
                            本系统基于 **Retrieval-Augmented Generation (RAG)** 技术，专为中国刑法领域设计。

                            ### 🚀 技术亮点
                            - **混合检索**: 结合关键词与语义向量检索，确保法条查找的准确性。
                            - **两阶段重排序**: 引入 LLM Reranker 对检索结果进行精细化排序。
                            - **幻觉检测**: 实时校验生成内容与引用法条的一致性，低置信度自动拒答。

                            ### 🛠️ 模块配置
                            | 模块 | 详情 |
                            |Data | 刑法法条 + CAIL2018 刑事案例库 |
                            |Embedding | BAAI/bge-m3 (1024 dim) |
                            |Vector DB | ChromaDB |
                            |LLM | Qwen2.5-7B-Instruct (SiliconFlow API) |
                            |Frontend | Gradio 5 + Custom CSS |
                            """)

        # Event Wiring
        def on_submit(message, history, citations_value, confidence_value):
            if _is_duplicate_submit(message, history):
                return history, "", citations_value, confidence_value
            answer, citations, confidence, new_history = chat_response(message, history)
            return new_history, "", citations, confidence

        msg_input.submit(
            fn=on_submit,
            inputs=[msg_input, chatbot, citations_display, confidence_display],
            outputs=[chatbot, msg_input, citations_display, confidence_display]
        )
        
        send_btn.click(
            fn=on_submit,
            inputs=[msg_input, chatbot, citations_display, confidence_display],
            outputs=[chatbot, msg_input, citations_display, confidence_display]
        )

        new_chat_btn.click(
            fn=clear_conversation,
            outputs=[chatbot, msg_input, citations_display, confidence_display]
        )

        search_btn.click(
            fn=search_documents,
            inputs=[search_input, search_k],
            outputs=search_results
        )

        show_sources.change(
            fn=lambda visible: gr.update(visible=visible),
            inputs=show_sources,
            outputs=side_panel
        )

        # Init Load
        def on_load():
            success, msg = initialize_engine()
            return msg

        app.load(fn=on_load, outputs=init_status)

    return app


if __name__ == "__main__":
    print("正在启动 Juris-RAG Web应用 (v2.3 UI Optimized)...")
    print("=" * 50)
    
    if not os.getenv("SILICONFLOW_API_KEY"):
        print("警告: 未检测到 SILICONFLOW_API_KEY 环境变量")
    
    app = create_app()
    
    share_enabled = os.getenv("GRADIO_SHARE", "false").lower() == "true"
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    
    launch_kwargs = {
        "server_name": server_name,
        "server_port": server_port,
        "share": share_enabled,
        "show_error": True,
        "favicon_path": None,
    }
    
    print(f"✅ 本地访问: http://{server_name}:{server_port}")
    
    if _USE_LAUNCH_THEME_CSS:
        launch_kwargs["theme"] = APP_THEME
        launch_kwargs["css"] = CUSTOM_CSS
        
    app.launch(**launch_kwargs)
