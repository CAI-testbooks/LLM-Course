# scripts/init_knowledge_base.py
"""
初始化知识库脚本
"""

import json
import time
from pathlib import Path
from typing import List, Dict

from src.data.document_processor import DocumentProcessor
from src.data.vector_store import VectorStoreManager
from src.utils.logger import setup_logger


def init_knowledge_base(config):
    """初始化知识库"""
    logger = setup_logger("init_kb", config.logging.file, config.logging.level)

    logger.info("开始初始化知识库...")
    start_time = time.time()

    try:
        # 1. 处理文档
        processor = DocumentProcessor(config)

        # 检查原始文档目录
        raw_path = Path(config.data.raw_documents_path)
        if not raw_path.exists():
            logger.error(f"原始文档目录不存在: {raw_path}")
            return

        # 处理文档
        logger.info(f"处理文档目录: {raw_path}")
        documents = processor.process_directory(str(raw_path))

        if not documents:
            logger.error("未找到任何文档")
            return

        logger.info(f"处理完成，共 {len(documents)} 个文档")

        # 2. 分块处理
        logger.info("开始文档分块...")
        chunks = processor.chunk_documents(documents)
        logger.info(f"分块完成，共 {len(chunks)} 个文本块")

        # 保存处理后的块
        processed_path = Path(config.data.processed_chunks_path)
        processed_path.parent.mkdir(parents=True, exist_ok=True)

        processor.save_chunks(chunks, str(processed_path))
        logger.info(f"已保存处理后的块到: {processed_path}")

        # 3. 创建向量数据库
        logger.info("创建向量数据库...")
        vector_store_manager = VectorStoreManager(config)
        vector_store = vector_store_manager.create_vector_store(chunks)

        # 4. 保存统计信息
        stats = {
            "documents": len(documents),
            "chunks": len(chunks),
            "avg_chunk_size": sum(len(c['content']) for c in chunks) / len(chunks) if chunks else 0,
            "processing_time": time.time() - start_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        stats_path = processed_path.parent / "statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        logger.info(f"知识库初始化完成，统计信息: {stats}")
        logger.info(f"总耗时: {time.time() - start_time:.2f}秒")

    except Exception as e:
        logger.error(f"初始化知识库失败: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    # 用于直接运行脚本
    from src.config.config_manager import RAGConfig

    config = RAGConfig.from_yaml("configs/config.yaml")
    init_knowledge_base(config)
