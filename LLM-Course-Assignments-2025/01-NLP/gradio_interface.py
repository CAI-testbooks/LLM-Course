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