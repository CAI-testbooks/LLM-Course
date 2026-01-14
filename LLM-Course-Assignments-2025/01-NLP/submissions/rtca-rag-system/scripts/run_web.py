# scripts/run_web.py
"""
启动Web界面脚本
"""

import time
from pathlib import Path

from src.api.gradio_app import GradioApp
from src.config.config_manager import RAGConfig
from src.data.vector_store import VectorStoreManager
from src.models.retriever import HybridRetriever
from src.rag.rag_system import RAGSystem
from src.utils.logger import setup_logger


def run_web_app(config):
    """运行Web应用"""
    logger = setup_logger("web_app", config.logging.file, config.logging.level)

    logger.info("启动Web应用...")
    logger.info(f"主机: {config.api.web_host}")
    logger.info(f"端口: {config.api.web_port}")

    try:
        # 1. 加载向量数据库
        logger.info("加载向量数据库...")
        vector_store_manager = VectorStoreManager(config)
        vector_store = vector_store_manager.load_vector_store()

        # 2. 加载文档块（用于稀疏检索）
        logger.info("加载文档块...")
        chunks_path = Path(config.data.processed_chunks_path)
        if not chunks_path.exists():
            logger.error(f"文档块文件不存在: {chunks_path}")
            raise FileNotFoundError(f"文档块文件不存在: {chunks_path}")

        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        logger.info(f"已加载 {len(chunks)} 个文档块")

        # 3. 创建检索器
        logger.info("创建检索器...")
        retriever = HybridRetriever(vector_store, chunks, config)

        # 4. 创建RAG系统
        logger.info("创建RAG系统...")
        rag_system = RAGSystem(config)
        rag_system.retriever = retriever

        # 5. 启动Gradio应用
        logger.info("启动Gradio界面...")
        app = GradioApp(rag_system)
        gradio_app = app.create_web_app()

        logger.info("Web应用启动完成!")

        # 启动服务
        gradio_app.launch(
            server_name=config.api.web_host,
            server_port=config.api.web_port,
            debug=config.api.web_debug,
            share=False  # 生产环境设置为False
        )

    except KeyboardInterrupt:
        logger.info("Web应用已停止")

    except Exception as e:
        logger.error(f"启动Web应用失败: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    # 用于直接运行脚本
    config = RAGConfig.from_yaml("configs/config.yaml")
    run_web_app(config)
