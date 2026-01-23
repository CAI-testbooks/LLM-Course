# src/rag_system.py
import re
import time
from typing import List, Dict, Optional
from .model_manager import QwenModelManager
from .retriever import HybridRetriever
from .config import RAGConfig


class RAGSystem:
    """RAG系统核心"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.model_manager = QwenModelManager(config)
        self.retriever = None
        self.conversation_history = []

        # 加载系统prompt
        self.system_prompt = self.load_system_prompt()

        # 不确定性检测关键词
        self.uncertain_keywords = [
            "我不确定", "我不知道", "无法确定", "没有找到", "未提及",
            "可能", "大概", "或许", "似乎", "应该"
        ]

    def load_system_prompt(self) -> str:
        """加载系统prompt"""
        return """你是一个航空标准RTCA DO-160G的专家助手。请基于提供的参考文档回答问题。
        回答要求：
        1. 准确引用文档中的具体章节和内容
        2. 如果问题超出文档范围，明确告知用户
        3. 对于不确定的内容，不要猜测，要承认不知道
        4. 回答要专业、准确、清晰
        
        当前文档：RTCA DO-160G 机载设备环境条件和试验程序"""

    def format_references(self, retrieved_docs: List[Dict]) -> str:
        """格式化引用信息"""
        references = []
        for i, doc in enumerate(retrieved_docs, 1):
            meta = doc['metadata']
            chapter_info = meta.get('chapter_info', {})
            references.append(
                f"[{i}] 来源：{meta.get('source', '未知')}，"
                f"章节：第{chapter_info.get('chapter', '未知')}章 {chapter_info.get('title', '')}，"
                f"页码：{meta.get('page', '未知')}，"
                f"相关性分数：{doc['score']:.3f}"
            )
        return "\n".join(references)

    def build_prompt(self, query: str, retrieved_docs: List[Dict]) -> str:
        """构建prompt"""
        # 构建上下文
        context = "\n\n".join([doc['content'] for doc in retrieved_docs])
        references = self.format_references(retrieved_docs)

        # 构建完整prompt
        prompt = f"""{self.system_prompt}

相关参考文档：
{context}

参考文档的详细来源信息：
{references}

用户问题：{query}

请基于以上参考文档回答问题，并在适当位置引用文档来源（如[1][2]）。如果文档中没有相关信息，请明确说明。

回答："""

        return prompt

    def detect_uncertainty(self, response: str) -> bool:
        """检测回答中的不确定性"""
        # 简单的关键词检测
        for keyword in self.uncertain_keywords:
            if keyword in response:
                return True

        # 检查是否有引用
        if not re.search(r'\[\d+\]', response):
            # 没有引用可能意味着不确定
            return True

        return False

    def retrieve_documents(self, query: str) -> List[Dict]:
        """检索相关文档"""
        if self.retriever is None:
            raise ValueError("检索器未初始化")
        return self.retriever.retrieve(query)

    def answer(self, query: str, conversation_id: str = None) -> Dict:
        """回答问题"""
        # 检索相关文档
        retrieved_docs = self.retrieve_documents(query)

        if not retrieved_docs:
            return {
                'answer': "抱歉，在文档中没有找到相关信息。",
                'references': [],
                'confidence': 0.0,
                'uncertain': True
            }

        # 构建prompt
        prompt = self.build_prompt(query, retrieved_docs)

        # 生成回答
        response = self.model_manager.generate(prompt)

        # 提取引用
        citations = re.findall(r'\[(\d+)\]', response)
        cited_docs = []
        for cite in citations:
            try:
                idx = int(cite) - 1
                if 0 <= idx < len(retrieved_docs):
                    cited_docs.append(retrieved_docs[idx])
            except:
                pass

        # 检测不确定性
        uncertain = self.detect_uncertainty(response)

        # 计算置信度（基于检索分数）
        avg_score = sum(doc['score']
                        for doc in cited_docs) / max(len(cited_docs), 1)
        confidence = min(avg_score * 10, 1.0)  # 归一化到0-1

        # 更新对话历史
        if conversation_id:
            self.update_conversation_history(conversation_id, query, response)

        return {
            'answer': response,
            'references': cited_docs,
            'confidence': confidence,
            'uncertain': uncertain,
            'retrieved_docs': retrieved_docs
        }

    def update_conversation_history(self, conv_id: str, query: str, response: str):
        """更新对话历史"""
        if conv_id not in self.conversation_history:
            self.conversation_history[conv_id] = []

        self.conversation_history[conv_id].extend([
            {"role": "user", "content": query},
            {"role": "assistant", "content": response}
        ])

        # 限制历史长度
        if len(self.conversation_history[conv_id]) > 10:
            self.conversation_history[conv_id] = self.conversation_history[conv_id][-10:]
