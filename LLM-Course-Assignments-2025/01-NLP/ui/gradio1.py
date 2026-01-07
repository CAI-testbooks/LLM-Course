#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG智能对话系统Gradio界面模块
提供用户友好的Web交互界面，包含RAG检索详情展示
"""

import os

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr
import tempfile
import shutil
import logging
from typing import List, Tuple, Optional, Dict, Any

from rag_system import RAGSystem

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGInterface:
    """
    RAG系统Gradio界面封装类
    """

    def __init__(self):
        """
        初始化界面
        """
        self.rag_system = None
        self.uploaded_files = []
        self.chat_history = []
        self.all_documents = []  # 记录所有文档

        # 创建临时目录存储上传的文件
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"临时目录创建: {self.temp_dir}")

    def upload_pdf(self, files) -> str:
        """
        处理PDF文件上传（支持增量添加到知识库）
        """
        if not files:
            return "请选择PDF文件上传"

        try:
            # 初始化文件列表
            if not hasattr(self, 'uploaded_files') or self.uploaded_files is None:
                self.uploaded_files = []

            new_files = []
            valid_files = []
            skipped_files = []

            for file in files:
                if file is None:
                    continue

                # 检查文件扩展名
                if not file.name.lower().endswith('.pdf'):
                    skipped_files.append(os.path.basename(file.name))
                    continue

                # 检查是否已经上传过
                file_basename = os.path.basename(file.name)
                temp_path = os.path.join(self.temp_dir, file_basename)

                if temp_path in self.uploaded_files:
                    skipped_files.append(f"{file_basename} (已存在)")
                    continue

                # 复制文件
                shutil.copy2(file.name, temp_path)

                self.uploaded_files.append(temp_path)
                new_files.append(temp_path)
                valid_files.append(file_basename)

            if not new_files:
                if skipped_files:
                    return f"没有新文件被添加。\n跳过: " + ", ".join(skipped_files)
                return "没有有效的PDF文件被上传"

            # 初始化RAG系统
            if self.rag_system is None:
                self.rag_system = RAGSystem()

            # 加载新文档
            new_documents = self.rag_system.load_pdf_documents(new_files)

            # 合并文档
            self.all_documents.extend(new_documents)

            if not self.all_documents:
                return "文档加载失败"

            # 重建向量库
            success = self.rag_system.build_vectorstore(self.all_documents)
            if not success:
                return "向量数据库构建失败"

            # 初始化问答链
            success = self.rag_system.init_qa_chain()
            if not success:
                return "问答系统初始化失败"

            status_msg = f"✅ 知识库更新成功！当前共 {len(self.all_documents)} 个文档片段。"
            if valid_files:
                status_msg += f"\n新增: {', '.join(valid_files)}"
            return status_msg

        except Exception as e:
            logger.error(f"文件处理失败: {str(e)}")
            return f"文件处理失败: {str(e)}"

    def clear_knowledge_base(self) -> str:
        """清空知识库"""
        try:
            self.uploaded_files = []
            self.all_documents = []
            if self.rag_system:
                self.rag_system.vectorstore = None
                self.rag_system.qa_chain = None
                self.rag_system.documents = []

            # 清理临时文件
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                os.makedirs(self.temp_dir)

            return "🗑️ 知识库已清空"
        except Exception as e:
            return f"清空失败: {str(e)}"

    def chat_with_rag(self, message: str, history: List[List[str]], temperature: float) -> Tuple[
        str, List[List[str]], List[Dict]]:
        """
        与RAG系统对话

        Returns:
            Tuple: (清空的输入框, 更新的历史记录, 检索到的上下文JSON)
        """
        if not message.strip():
            return "", history, []

        # 自动初始化检查
        if self.rag_system is None:
            try:
                self.rag_system = RAGSystem()
                # 尝试加载默认数据
                if os.path.exists("data/medical.json"):
                    docs = self.rag_system.load_medical_data("data/medical.json")
                    if docs:
                        self.rag_system.build_vectorstore(docs)
                self.rag_system.init_qa_chain(temperature)
            except Exception as e:
                err = f"系统初始化失败: {str(e)}"
                history.append([message, err])
                return "", history, []

        try:
            # 更新温度
            if hasattr(self.rag_system, 'llm') and self.rag_system.llm:
                self.rag_system.llm.temperature = temperature

            # === 核心调用 ===
            result = self.rag_system.ask_question(message)

            retrieved_context = []  # 用于前端展示的结构化数据

            if result["success"]:
                answer = result["answer"]
                source_docs = result.get("source_documents", [])

                # 1. 处理前端展示的引用文本
                if source_docs:
                    answer += "\n\n📚 **参考来源:**\n"
                    # 去重逻辑
                    seen_content = set()
                    idx = 1
                    for doc in source_docs:
                        # 提取内容摘要用于去重
                        content_sig = doc.get('content', '')[:50]
                        if content_sig not in seen_content:
                            seen_content.add(content_sig)
                            # 获取元数据
                            meta = doc.get('metadata', {})
                            source_name = meta.get('source', '未知来源')
                            if 'original_question' in meta:
                                source_name += f" - {meta['original_question']}"

                            answer += f"[{idx}] {source_name}\n"
                            idx += 1

                    # 2. 准备要在前端 JSON 面板展示的完整数据
                    for i, doc in enumerate(source_docs):
                        retrieved_context.append({
                            "rank": i + 1,
                            "content": doc.get("content", ""),
                            "metadata": doc.get("metadata", {}),
                            "score": doc.get("metadata", {}).get("rerank_score", "N/A")  # 如果有重排序分数
                        })
                else:
                    retrieved_context = [{"info": "未检索到相关文档，直接使用模型回答"}]

                # 添加响应时间
                if "response_time" in result:
                    answer += f"\n\n⏱️ 耗时: {result['response_time']:.2f}s"
            else:
                answer = f"处理失败: {result.get('error', '未知错误')}"
                retrieved_context = [{"error": str(result.get('error'))}]

            history.append([message, answer])
            return "", history, retrieved_context

        except Exception as e:
            error_response = f"发生错误: {str(e)}"
            logger.error(error_response)
            history.append([message, error_response])
            return "", history, [{"error": str(e)}]

    def clear_chat(self) -> Tuple[List, List]:
        """清空对话"""
        if self.rag_system:
            self.rag_system.clear_memory()
        self.chat_history = []
        return [], []  # 清空 history 和 retrieval_display

    def create_interface(self) -> gr.Blocks:
        """创建Gradio界面布局"""

        custom_css = """
        .gradio-container { max-width: 1400px !important; }
        .chat-window { height: 600px !important; }
        """

        with gr.Blocks(css=custom_css, title="医疗领域 RAG 问答系统") as interface:
            gr.Markdown("# 🏥 医疗领域特定 RAG 问答系统")

            with gr.Row():
                # === 左侧边栏：设置与状态 ===
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("### ⚙️ 系统设置")
                        temperature_slider = gr.Slider(0.1, 1.0, value=0.7, step=0.1, label="温度 (创造力)")

                        gr.Markdown("### 📁 知识库管理")
                        file_upload = gr.File(label="上传PDF文档", file_count="multiple", file_types=[".pdf"])
                        upload_btn = gr.Button("📥 处理并加载文档", variant="secondary")
                        upload_status = gr.Textbox(label="状态日志", lines=3, interactive=False)

                        clear_kb_btn = gr.Button("🗑️ 清空知识库")

                # === 中间：对话区域 ===
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        label="对话历史",
                        elem_classes=["chat-window"],
                        show_copy_button=True,
                        avatar_images=(None, "🤖")  # 用户头像默认，机器人头像
                    )

                    with gr.Row():
                        msg_input = gr.Textbox(
                            label="请输入您的医疗问题",
                            placeholder="例如：感冒了嗓子疼怎么办？ / 糖尿病饮食禁忌？",
                            scale=4,
                            lines=2
                        )
                        send_btn = gr.Button("🚀 发送", variant="primary", scale=1, size="lg")

                    clear_chat_btn = gr.Button("🧹 清空对话历史", size="sm")

                # === 右侧（或下方）：检索详情展示 (新增功能) ===
                with gr.Column(scale=1):
                    gr.Markdown("### 🔍 RAG 检索透视")
                    gr.Markdown("这里展示系统检索到的原始文档片段，用于验证回答依据。")

                    # 使用 JSON 组件展示结构化的检索结果
                    retrieval_display = gr.JSON(
                        label="当前问题的检索上下文 (Retrieved Context)",
                        value=[],
                        open=True
                    )

            # === 事件绑定 ===

            # 发送消息事件
            # 注意：outputs 增加了 retrieval_display
            send_btn.click(
                fn=self.chat_with_rag,
                inputs=[msg_input, chatbot, temperature_slider],
                outputs=[msg_input, chatbot, retrieval_display]
            )

            msg_input.submit(
                fn=self.chat_with_rag,
                inputs=[msg_input, chatbot, temperature_slider],
                outputs=[msg_input, chatbot, retrieval_display]
            )

            # 清空对话
            clear_chat_btn.click(
                fn=self.clear_chat,
                outputs=[chatbot, retrieval_display]
            )

            # 文件上传
            upload_btn.click(
                fn=self.upload_pdf,
                inputs=[file_upload],
                outputs=[upload_status]
            )

            clear_kb_btn.click(
                fn=self.clear_knowledge_base,
                outputs=[upload_status]
            )

        return interface


def main():
    rag_interface = RAGInterface()
    rag_interface.create_interface().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )


if __name__ == "__main__":
    main()