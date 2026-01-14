# src/document_processor.py
import os
import re
import json
from typing import List, Dict
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .config import RAGConfig


class DocumentProcessor:
    """文档处理器"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", "、", " "]
        )

    def load_pdf(self, file_path: str) -> List[Dict]:
        """加载PDF文档"""
        documents = []
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    # 提取章节信息
                    chapter_match = re.search(r'第(\d+)章\s+(.+)', text[:100])
                    chapter_info = {
                        'chapter': chapter_match.group(1) if chapter_match else str(page_num),
                        'title': chapter_match.group(2) if chapter_match else f"第{page_num}页",
                        'page': page_num
                    }

                    documents.append({
                        'content': text,
                        'metadata': {
                            'source': os.path.basename(file_path),
                            'page': page_num,
                            'chapter_info': chapter_info
                        }
                    })
        return documents

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """文档分块"""
        chunks = []
        for doc in documents:
            text_chunks = self.text_splitter.split_text(doc['content'])
            for i, chunk in enumerate(text_chunks):
                chunks.append({
                    'content': chunk,
                    'metadata': {
                        **doc['metadata'],
                        'chunk_id': i,
                        'start_char': i * self.config.chunk_size
                    }
                })
        return chunks

    def process_directory(self, dir_path: str) -> List[Dict]:
        """处理整个目录的文档"""
        all_chunks = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.pdf'):
                    file_path = os.path.join(root, file)
                    documents = self.load_pdf(file_path)
                    chunks = self.chunk_documents(documents)
                    all_chunks.extend(chunks)

        # 保存处理后的文档
        self.save_chunks(all_chunks, os.path.join(
            self.config.knowledge_base_path, "processed_chunks.json"))
        return all_chunks

    @staticmethod
    def save_chunks(chunks: List[Dict], output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
