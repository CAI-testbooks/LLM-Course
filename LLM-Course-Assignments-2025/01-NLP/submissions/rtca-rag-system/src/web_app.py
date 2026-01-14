# src/web_app.py
import gradio as gr
import time
from .rag_system import RAGSystem


class GradioApp:
    """Gradio Web应用"""

    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system
        self.conversations = {}  # conversation_id -> history

    def chat_interface(self, message: str, history: list, conversation_id: str):
        """聊天界面"""
        if not conversation_id:
            conversation_id = str(int(time.time()))

        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []

        # 获取回答
        result = self.rag_system.answer(message, conversation_id)

        # 格式化回答
        response = result['answer']
        if result['references']:
            response += "\n\n**参考来源：**\n"
            for i, ref in enumerate(result['references'], 1):
                meta = ref['metadata']
                response += f"{i}. {meta.get('source', '未知')} - 第{meta.get('page', '未知')}页\n"

        if result['uncertain']:
            response = "⚠️ **注意：** 这个回答可能不完全准确，建议核实官方文档。\n\n" + response

        # 更新历史
        self.conversations[conversation_id].append((message, response))

        return "", history + [(message, response)]

    def create_web_app(self):
        """创建Web应用"""
        with gr.Blocks(title="RTCA DO-160G专家助手", theme=gr.themes.Soft()) as app:
            gr.Markdown("# 🛩️ RTCA DO-160G专家助手")
            gr.Markdown("基于Qwen-2.5的航空标准文档智能问答系统")

            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(height=600)
                    msg = gr.Textbox(
                        label="请输入您的问题",
                        placeholder="例如：第4章的温度试验要求是什么？",
                        lines=2
                    )
                    with gr.Row():
                        submit_btn = gr.Button("发送", variant="primary")
                        clear_btn = gr.Button("清空对话")

                    conv_id = gr.Textbox(
                        label="会话ID（可选）",
                        placeholder="留空将创建新会话",
                        lines=1
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### 📊 系统信息")
                    confidence_bar = gr.Label("置信度: 待计算")
                    retrieval_stats = gr.Label("检索文档数: 0")
                    model_info = gr.Label(
                        f"模型: {self.rag_system.config.model_name}")

                    gr.Markdown("### ⚙️ 设置")
                    top_k_slider = gr.Slider(
                        minimum=1, maximum=10, value=5, step=1,
                        label="检索文档数量"
                    )
                    temp_slider = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.7, step=0.1,
                        label="生成温度"
                    )

                    gr.Markdown("### 📈 性能指标")
                    latency_display = gr.Label("响应时间: -")

            # 事件处理
            msg.submit(
                self.chat_interface,
                [msg, chatbot, conv_id],
                [msg, chatbot]
            )

            submit_btn.click(
                self.chat_interface,
                [msg, chatbot, conv_id],
                [msg, chatbot]
            )

            clear_btn.click(lambda: None, None, chatbot, queue=False)

            # 更新设置
            def update_settings(top_k, temperature):
                self.rag_system.config.top_k = int(top_k)
                self.rag_system.config.temperature = temperature
                return "设置已更新"

            top_k_slider.change(
                update_settings, [top_k_slider, temp_slider], [])
            temp_slider.change(update_settings, [
                               top_k_slider, temp_slider], [])

        return app
