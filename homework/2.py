"""
文件名：attachment14_enhanced_qa.py
版本：2.1（增强版）
描述：国际民航组织附件14第I卷智能问答系统 - 增强版
功能：多轮对话、章节目录提示、长上下文处理、精确引用、评估功能
作者：航空法规AI助手
日期：2024年1月
"""

import os
import re
import json
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# ==================== 导入外部库 ====================
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("警告: openai库未安装，AI功能将受限")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("警告: faiss库未安装，将使用简单检索")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("警告: sentence-transformers库未安装，将使用简单关键词检索")

# ==================== 数据类定义 ====================

@dataclass
class DocumentChunk:
    """
    文档分块数据类
    用于存储手册的分块内容及其元数据
    """
    id: str  # 块唯一标识符
    text: str  # 块文本内容
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据（章节、页码等）
    chapter_path: str = ""  # 章节路径（如：第1章>1.1>1.1.1）
    embedding: Optional[np.ndarray] = None  # 文本向量嵌入
    tokens: int = 0  # Token数量

    def __post_init__(self):
        """初始化后计算token数量"""
        self.tokens = len(self.text.split()) * 1.3  # 近似估计

@dataclass
class ConversationTurn:
    """
    对话轮次数据类
    记录单轮对话的完整信息
    """
    role: str  # "user" 或 "assistant"
    content: str  # 对话内容
    timestamp: datetime = field(default_factory=datetime.now)  # 时间戳
    citations: List[Dict] = field(default_factory=list)  # 引用来源
    confidence: float = 1.0  # 置信度分数
    query_used: Optional[str] = None  # 实际使用的查询（用于调试）
    chapter_suggestions: List[str] = field(default_factory=list)  # 章节目录建议

@dataclass
class ChapterInfo:
    """
    章节信息数据类
    存储章节结构信息
    """
    id: str  # 章节ID（如：1, 2.1, 3.2.1）
    title: str  # 章节标题
    level: int  # 章节级别（1-4）
    parent_id: Optional[str] = None  # 父章节ID
    content_summary: str = ""  # 内容摘要
    start_position: int = 0  # 在文档中的起始位置
    end_position: int = 0  # 在文档中的结束位置

@dataclass
class SearchResult:
    """
    搜索结果数据类
    封装检索到的相关信息
    """
    chunk: DocumentChunk  # 文档块
    score: float  # 相关性分数
    rank: int  # 排名

@dataclass
class QAEvaluationMetrics:
    """
    QA系统评估指标数据类
    用于量化系统性能
    """
    accuracy: float = 0.0  # 准确性
    citation_f1: float = 0.0  # 引用F1分数
    hallucination_rate: float = 0.0  # 幻觉率
    response_time: float = 0.0  # 响应时间（秒）
    user_satisfaction: float = 0.0  # 用户满意度

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'accuracy': self.accuracy,
            'citation_f1': self.citation_f1,
            'hallucination_rate': self.hallucination_rate,
            'response_time': self.response_time,
            'user_satisfaction': self.user_satisfaction
        }

# ==================== 章节管理器 ====================

class ChapterManager:
    """
    章节管理器：处理手册的章节结构和目录
    """

    def __init__(self):
        self.chapters: Dict[str, ChapterInfo] = {}  # 所有章节
        self.chapter_tree: Dict[str, List[str]] = defaultdict(list)  # 章节树结构
        self.toc_printed = False  # 是否已打印目录

    def parse_chapters(self, content: str) -> Dict[str, ChapterInfo]:
        """
        解析文档中的章节结构

        Args:
            content: 文档内容

        Returns:
            章节信息字典
        """
        print("📚 解析章节目录...")

        # 章节标题的正则表达式
        # 支持：## 第1章 总则, ### 1.1 定义, #### 1.1.1 某个概念
        patterns = [
            (r'^##\s+(第[一二三四五六七八九十\d]+章[^\n]*)', 1),  # 章
            (r'^###\s+(\d+\.\d+[^\n]*)', 2),  # 节
            (r'^####\s+(\d+\.\d+\.\d+[^\n]*)', 3),  # 小节
            (r'^#####\s+(\d+\.\d+\.\d+\.\d+[^\n]*)', 4),  # 小小节
        ]

        lines = content.split('\n')
        current_chapter_id = None
        current_level = 0
        chapter_content = []

        # 查找所有章节标题
        for i, line in enumerate(lines):
            for pattern, level in patterns:
                match = re.match(pattern, line.strip())
                if match:
                    title = match.group(1).strip()

                    # 提取章节ID
                    if level == 1:
                        # 章：提取数字
                        chapter_match = re.search(r'第([一二三四五六七八九十\d]+)章', title)
                        if chapter_match:
                            chapter_num = chapter_match.group(1)
                            # 中文数字转阿拉伯数字
                            cn_to_num = {'一':1,'二':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9,'十':10}
                            if chapter_num in cn_to_num:
                                chapter_id = str(cn_to_num[chapter_num])
                            else:
                                chapter_id = chapter_num
                        else:
                            chapter_id = str(len(self.chapters) + 1)

                    elif level >= 2:
                        # 节/小节：提取数字编号
                        num_match = re.search(r'^(\d+(\.\d+)*)', title)
                        if num_match:
                            chapter_id = num_match.group(1)
                        else:
                            chapter_id = f"{current_chapter_id}.{len([c for c in self.chapters.values() if c.parent_id == current_chapter_id]) + 1}"

                    # 创建章节信息
                    chapter_info = ChapterInfo(
                        id=chapter_id,
                        title=title,
                        level=level,
                        parent_id=current_chapter_id if level > 1 else None,
                        start_position=i
                    )

                    # 保存章节信息
                    self.chapters[chapter_id] = chapter_info

                    # 构建章节树
                    if level == 1:
                        self.chapter_tree['root'].append(chapter_id)
                    elif current_chapter_id:
                        self.chapter_tree[current_chapter_id].append(chapter_id)

                    # 更新当前章节
                    current_chapter_id = chapter_id
                    current_level = level

                    # 结束上一章的内容收集
                    if chapter_content and i > 0:
                        prev_chapter_id = list(self.chapters.keys())[-2] if len(self.chapters) > 1 else None
                        if prev_chapter_id and prev_chapter_id in self.chapters:
                            end_pos = i - 1
                            # 收集该章节的内容（最多10行）
                            content_lines = []
                            for j in range(self.chapters[prev_chapter_id].start_position + 1, min(end_pos, self.chapters[prev_chapter_id].start_position + 11)):
                                if j < len(lines):
                                    content_lines.append(lines[j].strip())

                            summary = ' '.join(content_lines[:5])[:200]
                            if len(summary) >= 200:
                                summary = summary[:197] + "..."
                            self.chapters[prev_chapter_id].content_summary = summary
                            self.chapters[prev_chapter_id].end_position = end_pos

                    chapter_content = []
                    break

        print(f"   解析到 {len(self.chapters)} 个章节")
        return self.chapters

    def get_toc(self, max_level: int = 3) -> str:
        """
        获取章节目录

        Args:
            max_level: 最大显示层级

        Returns:
            目录文本
        """
        if not self.chapters:
            return "尚未解析章节目录"

        toc_lines = ["📖 《附件14第I卷》章节目录", "="*60]

        def build_toc_recursive(parent_id: str, indent: int = 0):
            if parent_id not in self.chapter_tree:
                return

            for chapter_id in sorted(self.chapter_tree[parent_id],
                                   key=lambda x: [int(part) if part.isdigit() else part for part in x.split('.')]):
                if chapter_id in self.chapters:
                    chapter = self.chapters[chapter_id]
                    if chapter.level <= max_level:
                        prefix = "  " * indent
                        if chapter.level == 1:
                            toc_lines.append(f"{prefix}📗 {chapter.title}")
                        elif chapter.level == 2:
                            toc_lines.append(f"{prefix}  📘 {chapter.title}")
                        elif chapter.level == 3:
                            toc_lines.append(f"{prefix}    📙 {chapter.title}")
                        else:
                            toc_lines.append(f"{prefix}      📓 {chapter.title}")

                        # 添加简要说明
                        if chapter.content_summary and indent < 2:
                            summary = chapter.content_summary
                            if len(summary) > 80:
                                summary = summary[:77] + "..."
                            toc_lines.append(f"{prefix}      💡 {summary}")

                        build_toc_recursive(chapter_id, indent + 1)

        build_toc_recursive('root')
        toc_lines.append("="*60)
        toc_lines.append("💡 提示: 输入 '查看第X章' 或 '第X章内容' 获取详细内容")

        return "\n".join(toc_lines)

    def get_chapter_content(self, chapter_ref: str, content: str) -> Optional[str]:
        """
        获取指定章节内容

        Args:
            chapter_ref: 章节引用（如：第1章、2.1、3.2.1）
            content: 文档内容

        Returns:
            章节内容
        """
        lines = content.split('\n')

        # 查找章节
        target_chapter = None

        # 尝试匹配章节ID
        for chapter_id, chapter in self.chapters.items():
            if chapter_id == chapter_ref or chapter.title == chapter_ref:
                target_chapter = chapter
                break

        # 尝试模糊匹配
        if not target_chapter:
            for chapter_id, chapter in self.chapters.items():
                if chapter_ref in chapter.title or chapter_ref.replace('第', '').replace('章', '') in chapter.title:
                    target_chapter = chapter
                    break

        if not target_chapter:
            return None

        # 提取章节内容
        start_line = target_chapter.start_position
        end_line = target_chapter.end_position if target_chapter.end_position > 0 else start_line + 100

        chapter_lines = []
        in_chapter = False
        current_level = target_chapter.level

        for i in range(start_line, min(end_line, len(lines))):
            line = lines[i].strip()

            # 检查是否进入下一章节
            if i > start_line:
                for pattern, level in [(r'^##', 1), (r'^###', 2), (r'^####', 3), (r'^#####', 4)]:
                    if re.match(pattern, line):
                        if level <= current_level:
                            # 遇到同级或更高级标题，结束
                            return '\n'.join(chapter_lines)
                        break

            if line:
                chapter_lines.append(line)

        return '\n'.join(chapter_lines[:200])  # 限制长度

    def find_relevant_chapters(self, query: str, top_n: int = 5) -> List[Tuple[str, str, float]]:
        """
        查找与查询相关的章节

        Args:
            query: 查询文本
            top_n: 返回章节数量

        Returns:
            相关章节列表（章节ID，标题，相关性分数）
        """
        if not self.chapters:
            return []

        query_lower = query.lower()
        results = []

        for chapter_id, chapter in self.chapters.items():
            score = 0

            # 标题匹配
            if chapter.title:
                title_lower = chapter.title.lower()
                if query_lower in title_lower:
                    score += 5.0

                # 部分匹配
                query_words = query_lower.split()
                for word in query_words:
                    if len(word) > 2 and word in title_lower:
                        score += 1.0

            # 内容摘要匹配
            if chapter.content_summary:
                summary_lower = chapter.content_summary.lower()
                if query_lower in summary_lower:
                    score += 3.0

            if score > 0:
                results.append((chapter_id, chapter.title, score))

        # 按分数排序
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_n]

    def print_toc_if_needed(self):
        """如果需要，打印章节目录"""
        if not self.toc_printed:
            print("\n" + "="*60)
            print("📖 正在为您加载《附件14第I卷》章节目录...")
            print("="*60)

            # 只打印前2级的目录
            toc = self.get_toc(max_level=2)
            print(toc[:1500] + "..." if len(toc) > 1500 else toc)

            self.toc_printed = True

# ==================== 核心系统类 ====================

class Attachment14EnhancedQA:
    """
    附件14手册增强版问答系统主类
    包含章节目录提示和增强交互功能
    """

    def __init__(self,
                 manual_path: str,
                 api_key: str = None,
                 api_base: str = "https://api.siliconflow.cn/v1",
                 model_name: str = "Qwen/Qwen2.5-72B-Instruct",
                 use_embedding: bool = True,
                 show_toc: bool = True):
        """
        初始化问答系统

        Args:
            manual_path: 手册文件路径
            api_key: 硅基流动API密钥
            api_base: API基础URL
            model_name: 使用的模型名称
            use_embedding: 是否使用向量嵌入
            show_toc: 是否显示章节目录
        """
        print("🚀 初始化附件14增强版问答系统...")

        # 配置API
        self.api_key = api_key or "sk-bdgrimfksplnwstzulxfsrdijhjqribunforxvknatzpjlui"
        self.api_base = api_base
        self.model_name = model_name

        # 系统配置
        self.config = self._load_config()
        self.manual_path = manual_path
        self.show_toc = show_toc

        # 状态跟踪
        self.conversations: Dict[str, List[ConversationTurn]] = defaultdict(list)
        self.evaluation_metrics = QAEvaluationMetrics()
        self.system_stats = {
            'total_queries': 0,
            'successful_answers': 0,
            'rejected_answers': 0,
            'chapter_queries': 0,
            'avg_response_time': 0.0
        }

        # 初始化核心组件
        self._initialize_components(use_embedding)

        print("✅ 系统初始化完成!")
        print(f"   模型: {model_name}")
        print(f"   检索模式: {'语义向量检索' if use_embedding else '关键词检索'}")
        print(f"   章节数量: {len(self.chapter_manager.chapters)}")
        print(f"   文档块数量: {len(self.document_chunks)}")

    def _load_config(self) -> Dict:
        """
        加载系统配置

        Returns:
            配置字典
        """
        return {
            # 对话配置
            'max_history_turns': 10,  # 最大历史对话轮次
            'max_context_tokens': 32000,  # 最大上下文token数
            'summary_threshold': 5,  # 超过此轮次开始摘要

            # 检索配置
            'top_k_chunks': 5,  # 检索返回的文档块数量
            'similarity_threshold': 0.7,  # 相似度阈值
            'max_chunk_size': 1000,  # 文档块最大字符数
            'chunk_overlap': 100,  # 文档块重叠字符数

            # 回答生成配置
            'confidence_threshold': 0.7,  # 置信度阈值，低于此值拒绝回答
            'temperature': 0.3,  # 生成温度
            'max_tokens': 1500,  # 生成最大token数

            # 引用配置
            'require_citations': True,  # 要求引用来源
            'citation_format': "【来源{index}】",  # 引用格式

            # 章节配置
            'max_chapter_suggestions': 3,  # 最大章节建议数
            'auto_chapter_suggest': True,  # 自动提供章节建议

            # 评估配置
            'enable_evaluation': True,  # 启用评估
            'evaluation_sample_size': 100  # 评估样本大小
        }

    def _initialize_components(self, use_embedding: bool):
        """
        初始化系统核心组件

        Args:
            use_embedding: 是否使用向量嵌入
        """
        # 1. 加载并处理手册
        print("📖 加载手册文档...")
        self.manual_content = self._load_manual_content()

        # 2. 初始化章节管理器
        print("📚 初始化章节管理器...")
        self.chapter_manager = ChapterManager()
        self.chapter_manager.parse_chapters(self.manual_content)

        # 3. 分割文档为块
        print("🔪 分割文档为块...")
        self.document_chunks = self._split_document_into_chunks()
        print(f"   共分割为 {len(self.document_chunks)} 个文档块")

        # 4. 初始化检索系统
        print("🔍 初始化检索系统...")
        if use_embedding and SENTENCE_TRANSFORMERS_AVAILABLE:
            self.retriever = VectorRetriever(self.document_chunks)
            self.retrieval_mode = "semantic"
        else:
            self.retriever = KeywordRetriever(self.document_chunks)
            self.retrieval_mode = "keyword"

        # 5. 初始化对话管理器
        print("💬 初始化对话管理器...")
        self.dialogue_manager = DialogueManager(self.config)

        # 6. 初始化AI生成器
        print("🤖 初始化AI生成器...")
        self.ai_generator = AIGenerator(
            api_key=self.api_key,
            api_base=self.api_base,
            model_name=self.model_name,
            config=self.config
        )

        # 7. 初始化评估器（如果启用）
        if self.config['enable_evaluation']:
            print("📊 初始化评估器...")
            self.evaluator = QAEvaluator(self.document_chunks)

        # 8. 显示章节目录
        if self.show_toc:
            self.chapter_manager.print_toc_if_needed()

    def _load_manual_content(self) -> str:
        """
        加载手册内容

        Returns:
            手册文本内容
        """
        try:
            with open(self.manual_path, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"   成功加载手册，大小: {len(content)} 字符")
            return content
        except Exception as e:
            print(f"❌ 加载手册失败: {e}")
            return ""

    def _split_document_into_chunks(self) -> List[DocumentChunk]:
        """
        将手册分割为适合处理的文档块

        Returns:
            文档块列表
        """
        chunks = []
        chunk_id = 0

        # 使用章节信息指导分割
        current_chapter_path = ""
        lines = self.manual_content.split('\n')

        current_chunk_text = ""
        current_metadata = {}

        for i, line in enumerate(lines):
            # 检查是否是章节标题
            is_chapter_title = False
            chapter_id = None

            for pattern, level in [(r'^##\s+(第[一二三四五六七八九十\d]+章[^\n]*)', 1),
                                 (r'^###\s+(\d+\.\d+[^\n]*)', 2),
                                 (r'^####\s+(\d+\.\d+\.\d+[^\n]*)', 3)]:
                match = re.match(pattern, line.strip())
                if match:
                    is_chapter_title = True
                    title = match.group(1).strip()

                    # 查找对应的章节信息
                    for cid, chapter in self.chapter_manager.chapters.items():
                        if chapter.title == title:
                            chapter_id = cid
                            # 构建章节路径
                            if level == 1:
                                current_chapter_path = f"第{chapter_id}章"
                            elif level == 2:
                                # 查找父章节
                                if chapter.parent_id:
                                    parent_chapter = self.chapter_manager.chapters.get(chapter.parent_id)
                                    if parent_chapter:
                                        current_chapter_path = f"第{parent_chapter.id}章>{chapter_id}"
                            elif level == 3:
                                # 查找父章节和祖父章节
                                if chapter.parent_id:
                                    parent_chapter = self.chapter_manager.chapters.get(chapter.parent_id)
                                    if parent_chapter and parent_chapter.parent_id:
                                        grand_parent = self.chapter_manager.chapters.get(parent_chapter.parent_id)
                                        if grand_parent:
                                            current_chapter_path = f"第{grand_parent.id}章>{parent_chapter.id}>{chapter_id}"
                            break
                    break

            # 如果遇到章节标题或当前块已满，创建新块
            if (is_chapter_title and current_chunk_text) or len(current_chunk_text) > self.config['max_chunk_size']:
                if current_chunk_text:
                    chunk = DocumentChunk(
                        id=f"chunk_{chunk_id}",
                        text=current_chunk_text,
                        metadata=current_metadata.copy(),
                        chapter_path=current_chapter_path
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                    current_chunk_text = ""

                # 如果是章节标题，将其作为新块的开始
                if is_chapter_title:
                    current_chunk_text = line + "\n"
                    current_metadata = {
                        'chapter_id': chapter_id,
                        'chapter_title': title,
                        'is_chapter_start': True
                    }
            else:
                # 添加到当前块
                current_chunk_text += line + "\n"
                if not current_metadata and not is_chapter_title:
                    current_metadata = {'section': 'content'}

        # 添加最后一个块
        if current_chunk_text:
            chunk = DocumentChunk(
                id=f"chunk_{chunk_id}",
                text=current_chunk_text,
                metadata=current_metadata.copy(),
                chapter_path=current_chapter_path
            )
            chunks.append(chunk)

        # 如果上面的方法没有得到足够的块，使用简单分割
        if len(chunks) < 10:
            print("   使用备用分割方法...")
            chunks = []
            for i in range(0, len(self.manual_content), self.config['max_chunk_size']):
                chunk_text = self.manual_content[i:i + self.config['max_chunk_size']]
                chunk = DocumentChunk(
                    id=f"chunk_{i//self.config['max_chunk_size']}",
                    text=chunk_text,
                    metadata={'chunk_index': i//self.config['max_chunk_size']},
                    chapter_path=""
                )
                chunks.append(chunk)

        return chunks

    def ask(self,
            question: str,
            session_id: str = "default",
            use_history: bool = True,
            require_citations: bool = True) -> Dict[str, Any]:
        """
        主问答接口：处理用户提问并返回答案

        Args:
            question: 用户问题
            session_id: 会话ID（用于多轮对话）
            use_history: 是否使用对话历史
            require_citations: 是否需要引用来源

        Returns:
            包含答案和元数据的字典
        """
        import time
        start_time = time.time()
        self.system_stats['total_queries'] += 1

        print(f"\n{'='*60}")
        print(f"📝 用户问题: {question}")
        print(f"💡 会话ID: {session_id}")

        # 检查是否是章节目录查询
        chapter_response = self._handle_chapter_query(question)
        if chapter_response:
            self.system_stats['chapter_queries'] += 1
            return chapter_response

        # 1. 准备查询（结合历史）
        enriched_query = self._prepare_query(question, session_id, use_history)

        # 2. 检索相关文档
        search_results = self.retriever.search(
            query=enriched_query,
            top_k=self.config['top_k_chunks']
        )

        # 3. 检查是否有足够相关信息
        if not search_results or search_results[0].score < self.config['similarity_threshold']:
            response = self._handle_no_relevant_info(question, search_results)
            self.system_stats['rejected_answers'] += 1
            return response

        # 4. 准备生成上下文
        context_parts = self._prepare_context_parts(search_results, require_citations)

        # 5. 生成答案
        ai_response = self.ai_generator.generate_answer(
            question=question,
            context_parts=context_parts,
            conversation_history=self.conversations[session_id][-self.config['max_history_turns']:] if use_history else [],
            require_citations=require_citations
        )

        # 6. 解析答案并提取引用
        parsed_answer = self._parse_ai_response(ai_response, search_results)

        # 7. 生成章节建议
        chapter_suggestions = self._generate_chapter_suggestions(question, parsed_answer)

        # 8. 更新对话历史
        user_turn = ConversationTurn(
            role="user",
            content=question,
            query_used=enriched_query
        )

        assistant_turn = ConversationTurn(
            role="assistant",
            content=parsed_answer['answer'],
            citations=parsed_answer['citations'],
            confidence=parsed_answer['confidence'],
            chapter_suggestions=chapter_suggestions
        )

        self.conversations[session_id].extend([user_turn, assistant_turn])

        # 9. 管理对话历史长度（防止过长）
        self._manage_conversation_length(session_id)

        # 10. 记录响应时间
        response_time = time.time() - start_time
        self.system_stats['avg_response_time'] = (
            self.system_stats['avg_response_time'] * (self.system_stats['total_queries'] - 1) + response_time
        ) / self.system_stats['total_queries']

        # 11. 构建响应
        response = {
            'answer': parsed_answer['answer'],
            'citations': parsed_answer['citations'],
            'confidence': parsed_answer['confidence'],
            'chapter_suggestions': chapter_suggestions,
            'search_results': [
                {
                    'id': result.chunk.id,
                    'score': float(result.score),
                    'text': result.chunk.text[:200] + "..." if len(result.chunk.text) > 200 else result.chunk.text,
                    'metadata': result.chunk.metadata,
                    'chapter_path': result.chunk.chapter_path
                }
                for result in search_results[:3]
            ],
            'response_time': response_time,
            'session_id': session_id,
            'turn_count': len(self.conversations[session_id]) // 2
        }

        # 12. 如果启用评估，记录此问答
        if self.config['enable_evaluation']:
            self.evaluator.record_interaction(question, response)

        self.system_stats['successful_answers'] += 1

        print(f"✅ 生成答案完成 (耗时: {response_time:.2f}s)")
        print(f"   置信度: {parsed_answer['confidence']:.2%}")
        print(f"   引用数量: {len(parsed_answer['citations'])}")
        if chapter_suggestions:
            print(f"   章节建议: {len(chapter_suggestions)} 个")

        return response

    def _handle_chapter_query(self, question: str) -> Optional[Dict]:
        """
        处理章节目录查询

        Args:
            question: 用户问题

        Returns:
            如果是章节查询返回响应，否则返回None
        """
        question_lower = question.lower()

        # 检查是否是目录查询
        toc_keywords = ['目录', '章节', '第几章', '有什么章', 'toc', 'content', '章节列表']
        if any(keyword in question_lower for keyword in toc_keywords):
            print("📚 识别为章节目录查询")

            # 获取详细程度
            detail_level = 2  # 默认显示2级
            if '详细' in question_lower or '全部' in question_lower:
                detail_level = 4
            elif '简要' in question_lower or '简略' in question_lower:
                detail_level = 1

            toc = self.chapter_manager.get_toc(max_level=detail_level)

            response = {
                'answer': toc,
                'citations': [],
                'confidence': 1.0,
                'chapter_suggestions': [],
                'search_results': [],
                'response_time': 0.1,
                'session_id': "chapter_query",
                'turn_count': 0,
                'is_chapter_query': True
            }
            return response

        # 检查是否是具体章节查询
        chapter_patterns = [
            r'第[一二三四五六七八九十\d]+章',
            r'查看第[一二三四五六七八九十\d]+章',
            r'第\d+章内容',
            r'\d+\.\d+(\.\d+)*节?'
        ]

        for pattern in chapter_patterns:
            matches = re.findall(pattern, question)
            if matches:
                chapter_ref = matches[0].replace('查看', '').replace('内容', '').strip()
                print(f"📚 识别为章节内容查询: {chapter_ref}")

                chapter_content = self.chapter_manager.get_chapter_content(chapter_ref, self.manual_content)

                if chapter_content:
                    # 查找对应的章节信息
                    chapter_info = None
                    for chapter_id, chapter in self.chapter_manager.chapters.items():
                        if chapter_id in chapter_ref or chapter_ref in chapter.title:
                            chapter_info = chapter
                            break

                    if chapter_info:
                        answer = f"📖 {chapter_info.title}\n{'='*60}\n{chapter_content[:2000]}"
                        if len(chapter_content) > 2000:
                            answer += f"\n...\n💡 内容过长，只显示前2000字符。如需完整内容，请指定更具体的节号。"
                    else:
                        answer = f"📖 {chapter_ref}\n{'='*60}\n{chapter_content[:2000]}"

                    response = {
                        'answer': answer,
                        'citations': [],
                        'confidence': 1.0,
                        'chapter_suggestions': [],
                        'search_results': [],
                        'response_time': 0.1,
                        'session_id': "chapter_query",
                        'turn_count': 0,
                        'is_chapter_query': True
                    }
                    return response

        return None

    def _prepare_query(self, question: str, session_id: str, use_history: bool) -> str:
        """
        准备检索查询（结合对话历史）

        Args:
            question: 原始问题
            session_id: 会话ID
            use_history: 是否使用历史

        Returns:
            增强后的查询
        """
        if not use_history or session_id not in self.conversations:
            return question

        # 获取最近的对话历史
        recent_history = self.conversations[session_id][-self.config['max_history_turns']:]

        if not recent_history:
            return question

        # 提取历史中的关键信息
        context_keywords = []
        for turn in recent_history[-3:]:  # 只看最近3轮
            if turn.role == "user":
                # 从用户问题中提取名词短语作为关键词
                nouns = re.findall(r'[\u4e00-\u9fa5]{2,5}跑道|[\u4e00-\u9fa5]{2,5}道面|[\u4e00-\u9fa5]{2,5}灯光', turn.content)
                context_keywords.extend(nouns)

        # 合并关键词
        if context_keywords:
            enhanced_query = question + " " + " ".join(set(context_keywords))
            print(f"   增强查询: {enhanced_query}")
            return enhanced_query

        return question

    def _prepare_context_parts(self, search_results: List[SearchResult], require_citations: bool) -> List[str]:
        """
        准备生成答案的上下文部分

        Args:
            search_results: 检索结果
            require_citations: 是否需要引用

        Returns:
            上下文文本列表
        """
        context_parts = []

        for i, result in enumerate(search_results):
            chunk = result.chunk
            context_text = f"[文档块 {i+1}, ID:{chunk.id}, 相关性:{result.score:.3f}"

            if chunk.chapter_path:
                context_text += f", 章节:{chunk.chapter_path}"
            context_text += "]\n"

            if require_citations:
                # 添加引用标记
                context_text += f"【来源{i+1}】{chunk.text}"
            else:
                context_text += chunk.text

            context_parts.append(context_text)

        return context_parts

    def _parse_ai_response(self, ai_response: Dict, search_results: List[SearchResult]) -> Dict:
        """
        解析AI响应，提取答案和引用

        Args:
            ai_response: AI响应字典
            search_results: 检索结果

        Returns:
            解析后的答案字典
        """
        answer_text = ai_response.get('content', '')
        confidence = ai_response.get('confidence', 0.8)

        # 提取引用标记
        citations = []
        citation_pattern = r'【来源(\d+)】'

        # 查找所有引用标记
        citation_matches = list(re.finditer(citation_pattern, answer_text))

        for match in citation_matches:
            source_index = int(match.group(1)) - 1  # 转换为0-based索引
            if 0 <= source_index < len(search_results):
                chunk = search_results[source_index].chunk
                citations.append({
                    'source_index': source_index,
                    'chunk_id': chunk.id,
                    'text': chunk.text[:500] + "..." if len(chunk.text) > 500 else chunk.text,
                    'metadata': chunk.metadata,
                    'chapter_path': chunk.chapter_path
                })

        # 移除引用标记，使答案更易读
        clean_answer = re.sub(citation_pattern, '', answer_text).strip()

        return {
            'answer': clean_answer,
            'citations': citations,
            'confidence': confidence,
            'raw_answer': answer_text
        }

    def _generate_chapter_suggestions(self, question: str, parsed_answer: Dict) -> List[str]:
        """
        生成章节建议

        Args:
            question: 用户问题
            parsed_answer: 解析后的答案

        Returns:
            章节建议列表
        """
        if not self.config['auto_chapter_suggest']:
            return []

        suggestions = []
        max_suggestions = self.config['max_chapter_suggestions']

        # 基于问题查找相关章节
        relevant_chapters = self.chapter_manager.find_relevant_chapters(question, top_n=max_suggestions*2)

        # 基于答案中的关键词查找相关章节
        answer_keywords = re.findall(r'[\u4e00-\u9fa5]{2,6}|[A-Z]{2,}', parsed_answer['answer'])
        for keyword in answer_keywords[:5]:  # 取前5个关键词
            if len(keyword) > 1:
                keyword_chapters = self.chapter_manager.find_relevant_chapters(keyword, top_n=2)
                relevant_chapters.extend(keyword_chapters)

        # 去重和排序
        unique_chapters = {}
        for chapter_id, title, score in relevant_chapters:
            if chapter_id not in unique_chapters or score > unique_chapters[chapter_id][1]:
                unique_chapters[chapter_id] = (title, score)

        # 转换为列表并排序
        sorted_chapters = sorted(unique_chapters.items(), key=lambda x: x[1][1], reverse=True)

        # 生成建议文本
        for i, (chapter_id, (title, score)) in enumerate(sorted_chapters[:max_suggestions]):
            # 获取章节信息
            chapter_info = self.chapter_manager.chapters.get(chapter_id)
            if chapter_info:
                # 构建建议
                if chapter_info.content_summary:
                    suggestion = f"📘 {title} - {chapter_info.content_summary[:80]}..."
                else:
                    suggestion = f"📘 {title}"

                # 添加导航提示
                if chapter_info.level == 1:
                    suggestion += f" (详见第{chapter_id}章)"
                elif chapter_info.level == 2:
                    parent_chapter = self.chapter_manager.chapters.get(chapter_info.parent_id) if chapter_info.parent_id else None
                    if parent_chapter:
                        suggestion += f" (详见第{parent_chapter.id}章{chapter_id}节)"

                suggestions.append(suggestion)

        return suggestions

    def _handle_no_relevant_info(self, question: str, search_results: List[SearchResult]) -> Dict:
        """
        处理没有相关信息的情况

        Args:
            question: 用户问题
            search_results: 检索结果（可能为空）

        Returns:
            拒绝回答的响应
        """
        # 尝试从问题中提取关键词，提供手动搜索建议
        keywords = re.findall(r'[\u4e00-\u9fa5]{2,6}', question)
        keyword_suggestions = keywords[:3]

        # 查找相关章节作为建议
        chapter_suggestions = []
        if keyword_suggestions:
            for keyword in keyword_suggestions:
                relevant_chapters = self.chapter_manager.find_relevant_chapters(keyword, top_n=1)
                for chapter_id, title, score in relevant_chapters:
                    if score > 0.5:  # 只添加相关性较高的章节
                        chapter_suggestions.append(f"📘 {title}")

        suggestion_text = ""
        if chapter_suggestions:
            suggestion_text = f"\n\n💡 相关章节建议:\n" + "\n".join(chapter_suggestions)
        elif keyword_suggestions:
            suggestion_text = f"\n\n💡 建议查阅手册中关于'{'、'.join(keyword_suggestions)}'的章节。"

        # 提供章节目录提示
        toc_hint = "\n\n📚 您可以输入'目录'查看完整章节目录，或'第X章'查看具体章节内容。"

        return {
            'answer': f"抱歉，根据《附件14》手册的现有内容，我无法找到关于此问题的确切依据。{suggestion_text}{toc_hint}",
            'citations': [],
            'confidence': 0.3,
            'chapter_suggestions': chapter_suggestions,
            'search_results': [],
            'response_time': 0.1,
            'session_id': "rejected",
            'turn_count': 0,
            'is_rejected': True
        }

    def _manage_conversation_length(self, session_id: str):
        """
        管理对话历史长度，防止过长

        Args:
            session_id: 会话ID
        """
        max_turns = self.config['max_history_turns'] * 2  # user + assistant 对

        if len(self.conversations[session_id]) > max_turns:
            # 保留最近的对话，但总结早期部分
            old_history = self.conversations[session_id][:-max_turns]
            recent_history = self.conversations[session_id][-max_turns:]

            # 创建早期历史的摘要
            summary = self._create_conversation_summary(old_history)

            # 用摘要替换早期历史
            summary_turn = ConversationTurn(
                role="system",
                content=f"【早期对话摘要】{summary}",
                citations=[]
            )

            self.conversations[session_id] = [summary_turn] + recent_history

            print(f"   对话历史已摘要，当前轮次: {len(self.conversations[session_id])}")

    def _create_conversation_summary(self, history: List[ConversationTurn]) -> str:
        """
        创建对话历史摘要

        Args:
            history: 对话历史

        Returns:
            摘要文本
        """
        topics = []
        decisions = []

        for turn in history:
            if turn.role == "user":
                # 提取主题
                text_lower = turn.content.lower()
                if any(word in text_lower for word in ['跑道', 'runway']):
                    topics.append('跑道')
                elif any(word in text_lower for word in ['灯光', 'light']):
                    topics.append('灯光')
                elif any(word in text_lower for word in ['道面', 'pavement']):
                    topics.append('道面')

            if turn.role == "assistant" and turn.confidence > 0.8:
                # 提取高置信度的结论
                if '必须' in turn.content or '应' in turn.content:
                    # 提取关键句子
                    sentences = re.split(r'[。！？]', turn.content)
                    for sent in sentences:
                        if len(sent) > 10 and ('必须' in sent or '应' in sent):
                            decisions.append(sent[:100])

        summary_parts = []
        if topics:
            summary_parts.append(f"讨论了以下主题：{', '.join(set(topics))}")

        if decisions:
            summary_parts.append(f"明确了以下要求：{'；'.join(decisions[:3])}")

        return "；".join(summary_parts) if summary_parts else "无重要信息摘要"

    def get_conversation_history(self, session_id: str = "default") -> List[Dict]:
        """
        获取对话历史

        Args:
            session_id: 会话ID

        Returns:
            对话历史列表
        """
        if session_id not in self.conversations:
            return []

        return [
            {
                'role': turn.role,
                'content': turn.content,
                'time': turn.timestamp.strftime("%H:%M:%S"),
                'citations': turn.citations,
                'confidence': turn.confidence,
                'chapter_suggestions': turn.chapter_suggestions
            }
            for turn in self.conversations[session_id]
        ]

    def get_toc(self, detail_level: int = 2) -> str:
        """
        获取章节目录

        Args:
            detail_level: 详细级别（1-4）

        Returns:
            目录文本
        """
        return self.chapter_manager.get_toc(max_level=detail_level)

    def get_chapter_content(self, chapter_ref: str, max_length: int = 2000) -> Optional[str]:
        """
        获取指定章节内容

        Args:
            chapter_ref: 章节引用（如：第1章、2.1、3.2.1）
            max_length: 最大返回长度

        Returns:
            章节内容（截断到指定长度）
        """
        content = self.chapter_manager.get_chapter_content(chapter_ref, self.manual_content)
        if content and len(content) > max_length:
            content = content[:max_length] + "\n..."
        return content

    def evaluate_system(self, test_questions: List[Dict] = None) -> QAEvaluationMetrics:
        """
        评估系统性能

        Args:
            test_questions: 测试问题列表

        Returns:
            评估指标
        """
        if not self.config['enable_evaluation']:
            print("评估功能未启用")
            return self.evaluation_metrics

        print("\n" + "="*60)
        print("📊 开始系统性能评估...")

        # 使用内置测试问题（如果没有提供）
        if test_questions is None:
            test_questions = self._create_default_test_questions()

        total_questions = len(test_questions)
        print(f"   测试问题数量: {total_questions}")

        # 运行测试
        correct_answers = 0
        total_citations = 0
        correct_citations = 0
        hallucination_count = 0
        total_response_time = 0

        for i, test in enumerate(test_questions, 1):
            question = test['question']
            expected_answer = test.get('expected_answer', '')
            expected_citations = test.get('expected_citations', [])

            print(f"\n   [{i}/{total_questions}] 测试问题: {question}")

            # 获取系统答案
            response = self.ask(question, session_id=f"test_{i}", use_history=False)

            # 评估准确性
            is_correct = self.evaluator.evaluate_accuracy(
                response['answer'],
                expected_answer
            )

            if is_correct:
                correct_answers += 1

            # 评估引用
            citation_metrics = self.evaluator.evaluate_citations(
                response['citations'],
                expected_citations
            )
            total_citations += citation_metrics['total_expected']
            correct_citations += citation_metrics['correct']

            # 检查幻觉
            if self.evaluator.detect_hallucination(response['answer'], response['citations']):
                hallucination_count += 1

            total_response_time += response['response_time']

        # 计算指标
        accuracy = correct_answers / total_questions if total_questions > 0 else 0

        precision = correct_citations / (correct_citations + (total_citations - correct_citations)) if total_citations > 0 else 0
        recall = correct_citations / total_citations if total_citations > 0 else 0
        citation_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        hallucination_rate = hallucination_count / total_questions if total_questions > 0 else 0
        avg_response_time = total_response_time / total_questions if total_questions > 0 else 0

        # 更新评估指标
        self.evaluation_metrics.accuracy = accuracy
        self.evaluation_metrics.citation_f1 = citation_f1
        self.evaluation_metrics.hallucination_rate = hallucination_rate
        self.evaluation_metrics.response_time = avg_response_time

        print("\n" + "="*60)
        print("📈 评估结果:")
        print(f"   准确性: {accuracy:.2%}")
        print(f"   引用F1分数: {citation_f1:.3f}")
        print(f"   幻觉率: {hallucination_rate:.2%}")
        print(f"   平均响应时间: {avg_response_time:.2f}秒")
        print("="*60)

        return self.evaluation_metrics

    def _create_default_test_questions(self) -> List[Dict]:
        """
        创建默认测试问题

        Returns:
            测试问题列表
        """
        return [
            {
                'question': '什么是跑道端安全区？',
                'expected_answer': '跑道端安全区是与升降带端相邻的一块区域',
                'expected_citations': ['跑道端安全区', 'RESA']
            },
            {
                'question': 'PCN是什么意思？',
                'expected_answer': '道面等级号，表示道面承载强度的编号',
                'expected_citations': ['PCN', '道面等级号']
            },
            {
                'question': '跑道的最小宽度是多少？',
                'expected_answer': '根据基准代码和外侧主起落架轮距确定',
                'expected_citations': ['跑道宽度', '最小宽度']
            },
            {
                'question': '目录',
                'expected_answer': '章节目录',
                'expected_citations': []
            }
        ]

    def get_system_stats(self) -> Dict:
        """
        获取系统统计信息

        Returns:
            系统统计字典
        """
        return {
            'total_queries': self.system_stats['total_queries'],
            'successful_answers': self.system_stats['successful_answers'],
            'rejected_answers': self.system_stats['rejected_answers'],
            'chapter_queries': self.system_stats['chapter_queries'],
            'success_rate': self.system_stats['successful_answers'] / self.system_stats['total_queries'] if self.system_stats['total_queries'] > 0 else 0,
            'avg_response_time': self.system_stats['avg_response_time'],
            'active_sessions': len(self.conversations),
            'retrieval_mode': self.retrieval_mode,
            'total_chapters': len(self.chapter_manager.chapters)
        }

    def clear_conversation(self, session_id: str = "default"):
        """
        清除指定会话的历史

        Args:
            session_id: 会话ID
        """
        if session_id in self.conversations:
            self.conversations[session_id].clear()
            print(f"🗑️  已清除会话 {session_id} 的历史记录")

# ==================== 检索器基类与实现 ====================

class BaseRetriever:
    """检索器基类"""

    def __init__(self, chunks: List[DocumentChunk]):
        self.chunks = chunks

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """搜索相关文档块（需子类实现）"""
        raise NotImplementedError

class KeywordRetriever(BaseRetriever):
    """关键词检索器（简单实现）"""

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        基于关键词匹配的检索

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            搜索结果列表
        """
        results = []
        query_terms = set(re.findall(r'[\u4e00-\u9fa5]{2,6}|[A-Z]{2,}', query))

        for chunk in self.chunks:
            score = 0
            chunk_text = chunk.text

            for term in query_terms:
                if term in chunk_text:
                    score += 1
                    # 增加精确匹配的权重
                    if f" {term} " in f" {chunk_text} ":
                        score += 0.5

            # 归一化分数
            if query_terms:
                score = score / (len(query_terms) * 1.5)

            if score > 0:
                results.append(SearchResult(chunk=chunk, score=min(score, 1.0), rank=len(results)))

        # 按分数排序
        results.sort(key=lambda x: x.score, reverse=True)

        return results[:top_k]

class VectorRetriever(BaseRetriever):
    """向量检索器（基于语义相似度）"""

    def __init__(self, chunks: List[DocumentChunk], model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        super().__init__(chunks)

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("⚠️  sentence-transformers不可用，回退到关键词检索")
            self.use_vector = False
            return

        print(f"   加载嵌入模型: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        self.use_vector = True

        # 为所有块生成嵌入
        self._generate_embeddings()

        # 创建FAISS索引（如果可用）
        if FAISS_AVAILABLE:
            self._create_faiss_index()

    def _generate_embeddings(self):
        """为所有文档块生成向量嵌入"""
        print("   生成文档块嵌入...")
        texts = [chunk.text for chunk in self.chunks]

        # 批量生成嵌入（提高效率）
        batch_size = 32
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.embedding_model.encode(batch_texts, show_progress_bar=False)
            all_embeddings.append(batch_embeddings)

        # 合并所有嵌入
        embeddings = np.vstack(all_embeddings)

        # 分配给文档块
        for i, chunk in enumerate(self.chunks):
            chunk.embedding = embeddings[i]

    def _create_faiss_index(self):
        """创建FAISS向量索引"""
        print("   创建FAISS向量索引...")

        # 收集所有嵌入
        embeddings = []
        valid_chunks = []

        for chunk in self.chunks:
            if chunk.embedding is not None:
                embeddings.append(chunk.embedding)
                valid_chunks.append(chunk)

        if not embeddings:
            print("⚠️  无有效嵌入，无法创建FAISS索引")
            return

        embeddings = np.array(embeddings).astype('float32')
        dimension = embeddings.shape[1]

        # 创建索引
        self.faiss_index = faiss.IndexFlatIP(dimension)  # 内积相似度
        self.faiss_index.add(embeddings)
        self.indexed_chunks = valid_chunks

        print(f"   FAISS索引创建完成，维度: {dimension}, 文档数: {len(valid_chunks)}")

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        基于向量相似度的检索

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            搜索结果列表
        """
        if not self.use_vector:
            # 回退到关键词检索
            simple_retriever = KeywordRetriever(self.chunks)
            return simple_retriever.search(query, top_k)

        # 生成查询向量
        query_embedding = self.embedding_model.encode([query])[0].reshape(1, -1).astype('float32')

        if FAISS_AVAILABLE and hasattr(self, 'faiss_index'):
            # 使用FAISS搜索
            scores, indices = self.faiss_index.search(query_embedding, min(top_k * 2, len(self.indexed_chunks)))

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and score > 0:
                    chunk = self.indexed_chunks[idx]
                    # 归一化分数（内积相似度可能在0-1之间）
                    normalized_score = min(float(score), 1.0)
                    results.append(SearchResult(chunk=chunk, score=normalized_score, rank=len(results)))

            return results[:top_k]
        else:
            # 使用简单向量相似度计算
            results = []

            for chunk in self.chunks:
                if chunk.embedding is not None:
                    # 计算余弦相似度
                    similarity = np.dot(query_embedding[0], chunk.embedding) / (
                        np.linalg.norm(query_embedding[0]) * np.linalg.norm(chunk.embedding) + 1e-10
                    )

                    if similarity > 0:
                        results.append(SearchResult(chunk=chunk, score=float(similarity), rank=len(results)))

            # 按相似度排序
            results.sort(key=lambda x: x.score, reverse=True)

            return results[:top_k]

# ==================== 对话管理器 ====================

class DialogueManager:
    """对话管理器：处理多轮对话逻辑"""

    def __init__(self, config: Dict):
        self.config = config
        self.conversation_states = {}

    def update_context(self, session_id: str, user_query: str, system_response: str):
        """
        更新对话上下文

        Args:
            session_id: 会话ID
            user_query: 用户查询
            system_response: 系统响应
        """
        if session_id not in self.conversation_states:
            self.conversation_states[session_id] = {
                'history': [],
                'topic': None,
                'question_count': 0,
                'chapter_references': []  # 引用的章节
            }

        state = self.conversation_states[session_id]
        state['history'].append({
            'user': user_query,
            'system': system_response,
            'timestamp': datetime.now()
        })
        state['question_count'] += 1

        # 限制历史长度
        max_history = self.config.get('max_history_turns', 10)
        if len(state['history']) > max_history:
            state['history'] = state['history'][-max_history:]

        # 更新当前主题
        self._update_topic(session_id, user_query)

    def _update_topic(self, session_id: str, query: str):
        """
        更新对话主题

        Args:
            session_id: 会话ID
            query: 用户查询
        """
        state = self.conversation_states[session_id]

        # 从查询中提取可能的主题
        topic_keywords = {
            '跑道': ['跑道', 'runway', '升降带', '跑道端'],
            '灯光': ['灯光', 'light', '目视', '进近灯'],
            '道面': ['道面', 'pavement', 'PCN', 'ACN'],
            '障碍物': ['障碍物', 'obstacle', '限制面'],
            '标志': ['标志', 'marking', '标记'],
            '章节': ['目录', '第几章', '章节', 'toc']
        }

        for topic, keywords in topic_keywords.items():
            if any(keyword in query for keyword in keywords):
                state['topic'] = topic
                break

    def get_conversation_summary(self, session_id: str) -> str:
        """
        获取对话摘要

        Args:
            session_id: 会话ID

        Returns:
            对话摘要
        """
        if session_id not in self.conversation_states:
            return "无对话历史"

        state = self.conversation_states[session_id]

        if not state['history']:
            return "对话刚刚开始"

        # 提取关键信息
        topics = []
        questions = []

        for turn in state['history'][-3:]:  # 最近3轮
            if 'user' in turn:
                user_q = turn['user']
                # 简单提取
                if len(user_q) > 5:
                    questions.append(user_q[:50] + "...")

        summary = f"对话主题: {state['topic'] or '未指定'}, 问题数量: {state['question_count']}"
        if questions:
            summary += f", 最近问题: {'; '.join(questions)}"

        return summary

# ==================== AI生成器 ====================

class AIGenerator:
    """AI答案生成器：调用大模型生成答案"""

    def __init__(self, api_key: str, api_base: str, model_name: str, config: Dict):
        self.api_key = api_key
        self.api_base = api_base
        self.model_name = model_name
        self.config = config

        # 配置OpenAI客户端
        if OPENAI_AVAILABLE:
            openai.api_key = self.api_key
            openai.api_base = self.api_base
        else:
            print("⚠️  OpenAI库不可用，AI生成功能受限")

    def generate_answer(self,
                       question: str,
                       context_parts: List[str],
                       conversation_history: List[ConversationTurn] = None,
                       require_citations: bool = True) -> Dict[str, Any]:
        """
        生成答案

        Args:
            question: 用户问题
            context_parts: 上下文文本列表
            conversation_history: 对话历史
            require_citations: 是否需要引用

        Returns:
            生成的答案字典
        """
        # 构建提示词
        prompt = self._build_prompt(
            question,
            context_parts,
            conversation_history,
            require_citations
        )

        # 检查token长度
        estimated_tokens = len(prompt.split()) * 1.3

        if estimated_tokens > self.config['max_context_tokens']:
            print(f"⚠️  提示词过长 ({estimated_tokens:.0f} tokens)，进行压缩...")
            prompt = self._compress_prompt(prompt, context_parts)

        # 调用API
        response = self._call_api(prompt)

        # 解析响应
        if isinstance(response, str):
            content = response
            confidence = self._estimate_confidence(content, context_parts)
        elif isinstance(response, dict):
            content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
            confidence = response.get('confidence', 0.8)
        else:
            content = "抱歉，生成答案时出现错误。"
            confidence = 0.3

        return {
            'content': content,
            'confidence': confidence,
            'prompt_tokens': estimated_tokens
        }

    def _build_prompt(self,
                     question: str,
                     context_parts: List[str],
                     conversation_history: List[ConversationTurn],
                     require_citations: bool) -> str:
        """
        构建提示词

        Args:
            question: 用户问题
            context_parts: 上下文文本列表
            conversation_history: 对话历史
            require_citations: 是否需要引用

        Returns:
            完整的提示词
        """
        # 系统指令
        system_instruction = """你是一名严谨的国际民航组织附件14专家。请严格根据提供的《附件14第I卷：机场设计与运行》原文回答问题。

必须遵守以下规则：
1. 答案必须完全基于提供的上下文，不得添加任何外部知识或个人观点。
2. 如果上下文没有相关信息，必须明确回答"根据提供的资料无法回答此问题"。
3. 保持专业、准确，使用规范的航空术语。
4. 答案应简洁明了，但需包含必要的技术细节。
"""

        if require_citations:
            system_instruction += """5. 对于每个关键事实、数据或标准，必须在答案中标注来源，使用格式【来源X】，其中X对应上下文中的文档块编号。
6. 在答案末尾，以"引用来源："开头列出所有引用的文档块摘要。"""

        # 添加上下文
        context_text = "\n\n".join(context_parts)

        # 添加对话历史（如果存在）
        history_text = ""
        if conversation_history and len(conversation_history) > 0:
            history_text = "\n\n【对话历史】\n"
            for turn in conversation_history[-3:]:  # 最近3轮
                if turn.role == "user":
                    history_text += f"用户: {turn.content}\n"
                elif turn.role == "assistant":
                    history_text += f"助手: {turn.content[:100]}...\n"

        # 构建完整提示词
        prompt = f"""{system_instruction}

{history_text}

【相关上下文】
{context_text}

【当前问题】
{question}

请生成专业的答案："""

        return prompt

    def _compress_prompt(self, prompt: str, context_parts: List[str]) -> str:
        """
        压缩提示词（当超过token限制时）

        Args:
            prompt: 原始提示词
            context_parts: 上下文文本列表

        Returns:
            压缩后的提示词
        """
        # 简化上下文：只保留每个块的前200字符
        compressed_context = []
        for i, part in enumerate(context_parts):
            # 提取块的前部分
            lines = part.split('\n')
            if lines:
                first_line = lines[0]
                if len(first_line) > 200:
                    compressed_part = first_line[:200] + "..."
                else:
                    compressed_part = part[:300] + "..." if len(part) > 300 else part

                compressed_context.append(compressed_part)

        # 重新构建提示词
        system_part = prompt.split("【相关上下文】")[0]
        question_part = "【当前问题】" + prompt.split("【当前问题】")[1] if "【当前问题】" in prompt else ""

        compressed_prompt = f"{system_part}\n\n【相关上下文】\n" + "\n\n".join(compressed_context[:3]) + f"\n\n{question_part}"

        print(f"   提示词已压缩: {len(prompt.split())} -> {len(compressed_prompt.split())} 词")

        return compressed_prompt

    def _call_api(self, prompt: str) -> Union[str, Dict]:
        """
        调用API生成答案

        Args:
            prompt: 提示词

        Returns:
            API响应
        """
        if not OPENAI_AVAILABLE:
            # 模拟响应（用于测试）
            return "这是模拟的AI回答。在实际部署中，需要安装openai库并配置有效的API密钥。"

        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是专业的国际民航组织附件14专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.get('temperature', 0.3),
                max_tokens=self.config.get('max_tokens', 1500),
                timeout=30
            )

            return response

        except Exception as e:
            print(f"❌ API调用失败: {e}")
            return f"抱歉，AI服务暂时不可用。错误: {str(e)}"

    def _estimate_confidence(self, answer: str, context_parts: List[str]) -> float:
        """
        估计答案置信度

        Args:
            answer: 生成的答案
            context_parts: 上下文文本列表

        Returns:
            置信度分数 (0.0-1.0)
        """
        confidence = 0.8  # 基础置信度

        # 检查答案是否包含不确定词汇
        uncertainty_words = ['可能', '也许', '大概', '不确定', '无法确认', '不清楚']
        if any(word in answer for word in uncertainty_words):
            confidence *= 0.7

        # 检查答案是否明确拒绝
        if '无法回答' in answer or '没有相关信息' in answer:
            confidence = 0.3

        # 检查是否包含引用
        if '【来源' in answer:
            confidence *= 1.1  # 有引用增加置信度

        # 检查答案长度
        if len(answer) < 50:
            confidence *= 0.8  # 过短答案可能不完整

        return min(max(confidence, 0.0), 1.0)  # 限制在0-1之间

# ==================== 评估器 ====================

class QAEvaluator:
    """QA系统评估器"""

    def __init__(self, document_chunks: List[DocumentChunk]):
        self.document_chunks = document_chunks
        self.interactions = []

    def record_interaction(self, question: str, response: Dict):
        """
        记录交互信息

        Args:
            question: 问题
            response: 响应
        """
        self.interactions.append({
            'question': question,
            'response': response,
            'timestamp': datetime.now()
        })

    def evaluate_accuracy(self, actual_answer: str, expected_answer: str) -> bool:
        """
        评估答案准确性

        Args:
            actual_answer: 实际答案
            expected_answer: 期望答案

        Returns:
            是否准确
        """
        # 简单字符串匹配（可扩展为更复杂的NLP评估）
        actual_lower = actual_answer.lower()
        expected_lower = expected_answer.lower()

        # 检查关键术语是否匹配
        if expected_answer:
            expected_terms = re.findall(r'[\u4e00-\u9fa5]{2,6}|[A-Z]{2,}', expected_answer)
            matched_terms = 0

            for term in expected_terms:
                if term.lower() in actual_lower:
                    matched_terms += 1

            accuracy_ratio = matched_terms / len(expected_terms) if expected_terms else 0

            return accuracy_ratio > 0.6  # 60%的术语匹配即视为正确

        return False  # 没有期望答案，无法评估

    def evaluate_citations(self, actual_citations: List[Dict], expected_citations: List[str]) -> Dict:
        """
        评估引用质量

        Args:
            actual_citations: 实际引用
            expected_citations: 期望引用（关键词列表）

        Returns:
            引用评估指标
        """
        if not expected_citations:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'correct': 0, 'total_expected': 0}

        # 提取实际引用中的文本
        actual_texts = []
        for citation in actual_citations:
            if 'text' in citation:
                actual_texts.append(citation['text'])

        # 计算匹配情况
        matched_citations = 0

        for expected in expected_citations:
            for actual in actual_texts:
                if expected.lower() in actual.lower():
                    matched_citations += 1
                    break

        precision = matched_citations / len(actual_citations) if actual_citations else 0
        recall = matched_citations / len(expected_citations) if expected_citations else 0

        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'correct': matched_citations,
            'total_expected': len(expected_citations)
        }

    def detect_hallucination(self, answer: str, citations: List[Dict]) -> bool:
        """
        检测幻觉（答案中的信息无法被引用支持）

        Args:
            answer: 答案文本
            citations: 引用列表

        Returns:
            是否存在幻觉
        """
        if not citations:
            # 没有引用，但答案声称有特定信息，可能为幻觉
            specific_claims = ['必须', '应', '不得', '禁止', '要求', '标准']
            if any(claim in answer for claim in specific_claims) and len(answer) > 100:
                return True
            return False

        # 检查答案中的具体数据是否在引用中
        # 这里实现简单的检查逻辑
        citation_texts = " ".join([c.get('text', '') for c in citations])

        # 提取答案中的数字和专有名词
        numbers = re.findall(r'\d+\.?\d*', answer)
        terms = re.findall(r'[A-Z]{2,}[\d]*|[\u4e00-\u9fa5]{2,6}', answer)

        # 检查这些元素是否在引用中出现
        missing_elements = 0

        for num in numbers[:5]:  # 检查前5个数字
            if num not in citation_texts:
                missing_elements += 1

        for term in terms[:10]:  # 检查前10个术语
            if len(term) > 2 and term not in citation_texts:
                missing_elements += 1

        # 如果缺失元素比例过高，可能存在幻觉
        total_elements = min(5, len(numbers)) + min(10, len(terms))
        hallucination_ratio = missing_elements / total_elements if total_elements > 0 else 0

        return hallucination_ratio > 0.5  # 超过50%的元素未在引用中出现

# ==================== 使用示例和主函数 ====================

def main_demo():
    """
    主演示函数：展示系统功能
    """
    print("="*70)
    print("国际民航组织附件14第I卷 - 增强版问答系统")
    print("版本: 2.1 (包含章节目录提示)")
    print("="*70)

    # 配置文件路径（请修改为实际路径）
    manual_path = input("请输入手册文件路径 (直接回车跳过使用模拟数据): ").strip()

    if not manual_path:
        print("⚠️  未提供手册路径，将使用模拟模式")
        manual_path = "模拟路径"

    # 初始化系统
    qa_system = Attachment14EnhancedQA(
        manual_path=manual_path,
        use_embedding=False,  # 使用关键词检索（避免依赖外部库）
        show_toc=True
    )

    # 演示章节目录功能
    print("\n" + "="*70)
    print("演示1: 章节目录功能")
    print("="*70)

    # 获取章节目录
    toc = qa_system.get_toc(detail_level=2)
    print(toc[:1000] + "..." if len(toc) > 1000 else toc)

    # 演示问答功能
    print("\n" + "="*70)
    print("演示2: 智能问答功能")
    print("="*70)

    test_questions = [
        "什么是跑道端安全区？",
        "PCN是什么意思？",
        "目录",  # 测试章节目录查询
        "第3章讲了什么？",
        "跑道宽度有哪些要求？"
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n[{i}] 问: {question}")
        response = qa_system.ask(question, session_id="demo_session")

        print(f"答: {response['answer'][:300]}...")

        if response.get('chapter_suggestions'):
            print(f"📚 章节建议:")
            for suggestion in response['chapter_suggestions']:
                print(f"   • {suggestion}")

        if response.get('citations'):
            print(f"📖 引用来源: {len(response['citations'])} 个")

    # 演示多轮对话
    print("\n" + "="*70)
    print("演示3: 多轮对话功能")
    print("="*70)

    multi_session = "multi_turn_demo"

    questions = [
        "跑道端安全区的作用是什么？",
        "它的最小尺寸是多少？",
        "如果安装了拦阻系统呢？"
    ]

    for i, q in enumerate(questions, 1):
        print(f"\n第{i}轮问: {q}")
        response = qa_system.ask(q, session_id=multi_session, use_history=True)
        print(f"答: {response['answer'][:200]}... (置信度: {response['confidence']:.2%})")

    # 查看对话历史
    print("\n对话历史:")
    history = qa_system.get_conversation_history(multi_session)
    for i, turn in enumerate(history, 1):
        role_icon = "👤" if turn['role'] == 'user' else "🤖"
        print(f"  {i}. {role_icon} [{turn['time']}] {turn['content'][:80]}...")

    # 系统统计
    print("\n" + "="*70)
    print("系统统计信息")
    print("="*70)

    stats = qa_system.get_system_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "="*70)
    print("✅ 演示完成!")
    print("="*70)

def interactive_mode():
    """
    交互模式：命令行交互问答
    """
    print("="*70)
    print("附件14手册 - 交互式问答模式 (增强版)")
    print("="*70)
    print("📚 可用命令:")
    print("  '目录' 或 'toc' - 查看章节目录")
    print("  '第X章' - 查看具体章节内容")
    print("  '历史' - 查看对话历史")
    print("  '统计' - 查看系统统计")
    print("  '清除' - 清除对话历史")
    print("  '退出' 或 'quit' - 结束对话")
    print("="*70)

    # 初始化系统
    manual_path = input("请输入手册文件路径 (直接回车使用默认路径): ").strip()

    if not manual_path:
        # 这里应该设置一个默认路径
        manual_path = "附件14手册路径"
        print(f"使用默认路径: {manual_path}")

    try:
        qa_system = Attachment14EnhancedQA(
            manual_path=manual_path,
            use_embedding=False,  # 使用关键词检索
            show_toc=True
        )
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        print("⚠️  将使用模拟模式继续...")
        # 创建模拟系统
        qa_system = type('obj', (object,), {
            'ask': lambda self, q, **kwargs: {
                'answer': f"模拟回答: {q}",
                'citations': [],
                'confidence': 0.8,
                'chapter_suggestions': ['📘 第1章 总则', '📘 第3章 物理特性'],
                'response_time': 0.5
            },
            'get_toc': lambda self, detail_level=2: "📖 模拟章节目录\n1. 第1章 总则\n2. 第2章 机场数据\n3. 第3章 物理特性",
            'get_conversation_history': lambda self, session_id="default": [],
            'get_system_stats': lambda self: {'total_queries': 0, 'success_rate': 0},
            'clear_conversation': lambda self, session_id="default": print("🗑️  对话历史已清除")
        })()

    session_id = "interactive_session"

    while True:
        try:
            print("\n" + "-"*50)
            question = input("💭 请输入问题或命令: ").strip()

            if not question:
                continue

            # 检查命令
            question_lower = question.lower()

            if question_lower in ['quit', '退出', 'exit']:
                print("👋 感谢使用，再见！")
                break
            elif question_lower in ['clear', '清除', '清空']:
                qa_system.clear_conversation(session_id)
                print("🗑️  对话历史已清除")
                continue
            elif question_lower in ['history', '历史', '对话历史']:
                history = qa_system.get_conversation_history(session_id)
                if not history:
                    print("📝 无对话历史")
                else:
                    print("\n📝 对话历史:")
                    for i, turn in enumerate(history, 1):
                        role_icon = "👤" if turn['role'] == 'user' else "🤖"
                        confidence_str = f" ({turn['confidence']:.0%})" if 'confidence' in turn else ""
                        print(f"  {i}. {role_icon}{confidence_str} {turn['content'][:80]}...")
                continue
            elif question_lower in ['stats', '统计', '状态']:
                stats = qa_system.get_system_stats()
                print("\n📊 系统统计:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                continue
            elif question_lower in ['toc', '目录', '章节目录']:
                detail_level = 2
                if '详细' in question:
                    detail_level = 4
                elif '简要' in question:
                    detail_level = 1

                toc = qa_system.get_toc(detail_level=detail_level)
                print(f"\n{toc}")
                continue

            # 处理问题
            print("⏳ 正在搜索和生成答案...")
            response = qa_system.ask(question, session_id=session_id)

            print(f"\n{'='*60}")
            print(f"✅ 答案 (置信度: {response['confidence']:.2%}):")
            print(f"{response['answer']}")

            # 显示章节建议（如果有）
            if response.get('chapter_suggestions'):
                print(f"\n📚 相关章节建议:")
                for suggestion in response['chapter_suggestions']:
                    print(f"  • {suggestion}")

            # 显示引用（如果有）
            if response.get('citations'):
                print(f"\n📖 引用来源 ({len(response['citations'])} 个):")
                for i, citation in enumerate(response['citations'], 1):
                    chapter_info = citation.get('chapter_path', '未知章节')
                    print(f"  {i}. 【{citation['chunk_id']}】{chapter_info}")
                    print(f"     原文: {citation['text'][:100]}...")

            # 显示响应时间
            print(f"\n⏱️  响应时间: {response['response_time']:.2f}秒")

            # 提示用户可以使用章节目录功能
            if '无法找到' in response['answer'] or response['confidence'] < 0.5:
                print(f"\n💡 提示: 可以尝试输入'目录'查看手册结构，或输入'第X章'查看具体内容")

        except KeyboardInterrupt:
            print("\n\n👋 用户中断，退出系统")
            break
        except Exception as e:
            print(f"❌ 处理问题时出错: {e}")
            print("💡 建议检查手册文件路径是否正确，或使用更简单的问题重试")

# ==================== 快速启动函数 ====================

def quick_start():
    """
    快速启动函数：简化系统启动流程
    """
    print("🚀 附件14问答系统 - 快速启动")
    print("="*50)

    print("请选择模式:")
    print("1. 交互模式 (命令行问答)")
    print("2. 演示模式 (查看系统功能)")
    print("3. 退出")

    choice = input("\n请选择 (1-3): ").strip()

    if choice == "1":
        interactive_mode()
    elif choice == "2":
        main_demo()
    elif choice == "3":
        print("👋 再见!")
    else:
        print("❌ 无效选择，请重新运行程序")

if __name__ == "__main__":
    # 直接运行交互模式
    interactive_mode()

    # 或者运行快速启动菜单
    # quick_start()