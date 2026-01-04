#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG智能对话系统Gradio界面模块
提供用户友好的Web交互界面
"""

import os

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
import logging

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

        # 创建临时目录存储上传的文件
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"临时目录创建: {self.temp_dir}")

    def upload_pdf(self, files) -> str:
        """
        处理PDF文件上传（支持增量添加到知识库）

        Args:
            files: 上传的文件列表

        Returns:
            str: 处理结果信息
        """
        if not files:
            return "请选择PDF文件上传"

        try:
            # 如果是第一次上传，初始化文件列表
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
                    logger.warning(f"跳过非PDF文件: {file.name}")
                    skipped_files.append(os.path.basename(file.name))
                    continue

                # 检查是否已经上传过该文件
                file_basename = os.path.basename(file.name)
                temp_path = os.path.join(self.temp_dir, file_basename)

                if temp_path in self.uploaded_files:
                    logger.info(f"文件已存在于知识库中: {file_basename}")
                    skipped_files.append(f"{file_basename} (已存在)")
                    continue

                # 复制文件到临时目录
                shutil.copy2(file.name, temp_path)

                self.uploaded_files.append(temp_path)
                new_files.append(temp_path)
                valid_files.append(file_basename)

            if not new_files:
                if skipped_files:
                    return f"没有新文件被添加。跳过的文件:\n" + "\n".join([f"• {name}" for name in skipped_files])
                else:
                    return "没有有效的PDF文件被上传"

            # 初始化RAG系统（如果还没有初始化）
            if self.rag_system is None:
                self.rag_system = RAGSystem()

            # 加载新文档
            new_documents = self.rag_system.load_pdf_documents(new_files)

            # 如果已有知识库，需要合并文档
            if hasattr(self, 'all_documents') and self.all_documents:
                self.all_documents.extend(new_documents)
                logger.info(f"向现有知识库添加 {len(new_documents)} 个新文档片段")
            else:
                self.all_documents = new_documents
                logger.info(f"创建新知识库，包含 {len(new_documents)} 个文档片段")

            documents = self.all_documents

            if not documents:
                return "PDF文件加载失败，请检查文件格式"

            # 重新构建向量数据库（包含所有文档）
            success = self.rag_system.build_vectorstore(documents)

            if not success:
                return "向量数据库构建失败"

            # 初始化QA链
            success = self.rag_system.init_qa_chain()

            if not success:
                return "问答系统初始化失败"

            # 构建状态消息
            total_files = len(self.uploaded_files)
            total_docs = len(documents)
            new_file_count = len(valid_files)

            status_msg = f"✅ 知识库更新成功！\n\n"
            status_msg += f"📊 知识库状态：\n"
            status_msg += f"• 总文件数：{total_files} 个PDF文件\n"
            status_msg += f"• 总文档片段：{total_docs} 个\n\n"

            if new_file_count > 0:
                status_msg += f"📁 本次新增文件 ({new_file_count} 个)：\n"
                status_msg += "\n".join([f"• {name}" for name in valid_files])

            if skipped_files:
                status_msg += f"\n\n⚠️ 跳过的文件：\n"
                status_msg += "\n".join([f"• {name}" for name in skipped_files])

            return status_msg

        except Exception as e:
            error_msg = f"文件处理失败: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def clear_knowledge_base(self) -> str:
        """
        清空知识库

        Returns:
            str: 清空结果信息
        """
        try:
            # 清空文件列表
            if hasattr(self, 'uploaded_files'):
                self.uploaded_files = []

            # 清空文档列表
            if hasattr(self, 'all_documents'):
                self.all_documents = []

            # 重置RAG系统
            if self.rag_system:
                self.rag_system.vectorstore = None
                self.rag_system.qa_chain = None
                self.rag_system.tfidf_embeddings = None
                self.rag_system.documents = []

            # 清理临时文件
            if os.path.exists(self.temp_dir):
                for file in os.listdir(self.temp_dir):
                    file_path = os.path.join(self.temp_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)

            logger.info("知识库已清空")
            return "🗑️ 知识库已清空，可以重新上传文件构建新的知识库。"

        except Exception as e:
            logger.error(f"清空知识库失败: {str(e)}")
            return f"清空失败: {str(e)}"

    def chat_with_rag(self, message: str, history: List[List[str]], temperature: float) -> Tuple[str, List[List[str]]]:
        """
        与RAG系统对话

        Args:
            message: 用户输入的消息
            history: 对话历史
            temperature: 生成温度

        Returns:
            Tuple: (空字符串, 更新后的对话历史)
        """
        if not message.strip():
            return "", history

        # 如果RAG系统未初始化，先初始化它
        if self.rag_system is None:
            try:
                self.rag_system = RAGSystem()
                # 初始化问答链（即使没有文档也可以工作）
                success = self.rag_system.init_qa_chain(temperature)
                if not success:
                    error_response = "RAG系统初始化失败，请检查配置"
                    history.append([message, error_response])
                    return "", history
            except Exception as e:
                error_response = f"RAG系统初始化失败: {str(e)}"
                history.append([message, error_response])
                return "", history

        # 如果问答链未初始化，先初始化它
        if not self.rag_system.qa_chain:
            try:
                success = self.rag_system.init_qa_chain(temperature)
                if not success:
                    error_response = "问答链初始化失败，请检查配置"
                    history.append([message, error_response])
                    return "", history
            except Exception as e:
                error_response = f"问答链初始化失败: {str(e)}"
                history.append([message, error_response])
                return "", history

        try:
            # 更新温度参数
            if hasattr(self.rag_system, 'llm') and self.rag_system.llm:
                self.rag_system.llm.temperature = temperature

            # 获取回答
            result = self.rag_system.ask_question(message)

            if result["success"]:
                # 构建回答，包含来源信息
                answer = result["answer"]

                # 添加模式信息
                mode_info = ""
                if "mode" in result:
                    if result["mode"] == "simple":
                        if result["source_documents"]:
                            mode_info = "\n\n🔍 **检索模式**: 基于已上传文档回答"
                        else:
                            mode_info = "\n\n🤖 **对话模式**: 基于大模型知识回答"
                    else:
                        mode_info = "\n\n📚 **知识库模式**: 基于向量检索回答"

                if result["source_documents"]:
                    answer += "\n\n📚 **参考来源:**\n"
                    # 去重文件名
                    unique_files = set()
                    file_references = []

                    for source in result["source_documents"]:
                        if 'metadata' in source and 'source' in source['metadata']:
                            file_name = os.path.basename(source['metadata']['source'])
                            if file_name not in unique_files:
                                unique_files.add(file_name)
                                file_references.append(file_name)
                        else:
                            # 对于没有文件信息的片段，仍然添加到引用中
                            content_snippet = f"文档片段: {source['content'][:100]}..."
                            if content_snippet not in file_references:
                                file_references.append(content_snippet)

                    # 显示去重后的引用
                    for i, ref in enumerate(file_references, 1):
                        answer += f"[{i}] {ref}\n"

                # 添加模式和响应时间信息
                answer += mode_info
                if "response_time" in result:
                    answer += f"\n⏱️ 响应时间: {result['response_time']:.2f}秒"

            else:
                answer = result["answer"]

            # 更新对话历史
            history.append([message, answer])

        except Exception as e:
            error_response = f"处理消息时出现错误: {str(e)}"
            logger.error(error_response)
            history.append([message, error_response])

        return "", history

    def clear_chat(self) -> List:
        """
        清空对话历史

        Returns:
            List: 空的对话历史
        """
        if self.rag_system:
            self.rag_system.clear_memory()

        self.chat_history = []
        return []

    def get_system_status(self) -> str:
        """
        获取系统状态信息

        Returns:
            str: 系统状态描述
        """
        if self.rag_system is None:
            return "❌ RAG系统未初始化"

        status_info = []
        status_info.append("✅ RAG系统已初始化")

        if self.rag_system.embeddings:
            status_info.append("✅ 嵌入模型已加载")

        if self.rag_system.vectorstore:
            status_info.append("✅ 向量数据库已构建")

        if self.rag_system.qa_chain:
            status_info.append("✅ 问答链已初始化")

        if self.uploaded_files:
            status_info.append(f"📁 已加载 {len(self.uploaded_files)} 个PDF文件")

        if self.rag_system.memory:
            memory_info = self.rag_system.get_memory_summary()
            status_info.append(f"💭 {memory_info}")

        return "\n".join(status_info)

    def create_interface(self) -> gr.Blocks:
        """
        创建Gradio界面

        Returns:
            gr.Blocks: Gradio界面对象
        """
        # 自定义CSS样式
        custom_css = """
        .gradio-container {
            max-width: 1200px !important;
            margin: auto !important;
        }
        .chat-container {
            height: 500px !important;
        }
        .upload-area {
            border: 2px dashed #ccc !important;
            border-radius: 10px !important;
            padding: 20px !important;
            text-align: center !important;
        }
        """

        with gr.Blocks(css=custom_css, title="RAG智能对话系统") as interface:
            # 标题和说明
            gr.Markdown(
                """
                # 🤖 基于大模型和RAG的医学知识问答系统

                基于检索增强生成(RAG)技术的智能对话系统，支持两种对话模式：

                ## 🤖 直接对话模式
                - 无需上传文档，直接与大模型对话
                - 基于模型训练知识回答问题

                ## 📚 知识库模式  
                - 上传PDF文档构建个人知识库
                - 基于文档内容进行精准回答

                ## 使用说明：
                1. 💬 **可直接开始对话** - 无需上传文档
                2. 📁 上传PDF文档（可选，用于构建知识库）
                3. ⚙️ 调整生成参数（可选）
                4. 🔄 可随时清空对话历史
                """
            )

            with gr.Row():
                # 左侧：文件上传和系统状态
                with gr.Column(scale=1):
                    gr.Markdown("### 📁 文档上传（可选）")

                    file_upload = gr.File(
                        label="选择PDF文件",
                        file_count="multiple",
                        file_types=[".pdf"],
                        elem_classes=["upload-area"]
                    )

                    upload_status = gr.Textbox(
                        label="上传状态",
                        interactive=False,
                        lines=5
                    )

                    with gr.Row():
                        upload_btn = gr.Button("🚀 处理文档", variant="primary", scale=2)
                        clear_kb_btn = gr.Button("🗑️ 清空知识库", variant="secondary", scale=1)

                    gr.Markdown("### ⚙️ 参数设置")

                    temperature_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="生成温度 (控制回答的创造性)",
                        info="较低值更保守，较高值更有创意"
                    )

                    gr.Markdown("### 📊 系统状态")

                    system_status = gr.Textbox(
                        label="当前状态",
                        value="✅ 系统已就绪，可直接开始对话\n💡 提示：上传PDF文档可启用知识库模式",
                        interactive=False,
                        lines=6
                    )

                    status_refresh_btn = gr.Button("🔄 刷新状态")

                # 右侧：对话界面
                with gr.Column(scale=2):
                    gr.Markdown("### 💬 智能对话")

                    chatbot = gr.Chatbot(
                        label="对话历史",
                        height=500,
                        elem_classes=["chat-container"]
                    )

                    with gr.Row():
                        msg_input = gr.Textbox(
                            label="输入您的问题",
                            placeholder="请输入您想了解的问题...",
                            scale=4
                        )

                        send_btn = gr.Button("📤 发送", variant="primary", scale=1)

                    with gr.Row():
                        clear_btn = gr.Button("🗑️ 清空对话", variant="secondary")

                    gr.Markdown(
                        """
                        ### 💡 使用提示：
                        - 支持多轮对话，系统会记住上下文
                        - 回答会显示参考的文档来源
                        - 可以询问文档中的具体内容
                        - 支持跨文档的综合性问题
                        """
                    )

            # 事件绑定
            upload_btn.click(
                fn=self.upload_pdf,
                inputs=[file_upload],
                outputs=[upload_status]
            ).then(
                fn=self.get_system_status,
                outputs=[system_status]
            )

            clear_kb_btn.click(
                fn=self.clear_knowledge_base,
                outputs=[upload_status]
            ).then(
                fn=self.get_system_status,
                outputs=[system_status]
            )

            send_btn.click(
                fn=self.chat_with_rag,
                inputs=[msg_input, chatbot, temperature_slider],
                outputs=[msg_input, chatbot]
            )

            msg_input.submit(
                fn=self.chat_with_rag,
                inputs=[msg_input, chatbot, temperature_slider],
                outputs=[msg_input, chatbot]
            )

            clear_btn.click(
                fn=self.clear_chat,
                outputs=[chatbot]
            ).then(
                fn=self.get_system_status,
                outputs=[system_status]
            )

            status_refresh_btn.click(
                fn=self.get_system_status,
                outputs=[system_status]
            )

            # 页面加载时更新状态
            interface.load(
                fn=self.get_system_status,
                outputs=[system_status]
            )

        return interface

    def launch(self, **kwargs):
        """
        启动界面

        Args:
            **kwargs: Gradio launch参数
        """
        interface = self.create_interface()

        # 默认启动参数
        default_kwargs = {
            "server_name": "0.0.0.0",
            "server_port": 7860,
            "share": False,
            "debug": True
        }

        # 合并用户参数
        launch_kwargs = {**default_kwargs, **kwargs}

        logger.info(f"启动Gradio界面，参数: {launch_kwargs}")

        try:
            interface.launch(**launch_kwargs)
        except Exception as e:
            logger.error(f"界面启动失败: {str(e)}")
            raise

    def __del__(self):
        """
        清理临时文件
        """
        try:
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info(f"临时目录已清理: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"清理临时目录失败: {str(e)}")


def main():
    """
    主函数
    """
    try:
        # 创建并启动界面
        rag_interface = RAGInterface()
        rag_interface.launch(
            share=False,  # 设置为True可以生成公共链接
            debug=True
        )
    except Exception as e:
        logger.error(f"程序启动失败: {str(e)}")
        raise


if __name__ == "__main__":
    main()