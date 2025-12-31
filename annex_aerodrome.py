import os
import json
import re
import hashlib
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass
import warnings
import sys
import time

warnings.filterwarnings('ignore')

# 设置硅基流动API
import openai
from openai import OpenAI

# 配置API
client = OpenAI(
    api_key="sk-bdgrimfksplnwstzulxfsrdijhjqribunforxvknatzpjlui",
    base_url="https://api.siliconflow.cn/v1"
)


@dataclass
class SearchResult:
    """搜索结果数据结构"""
    content: str
    chapter: str
    section: str
    confidence: float
    source_page: Optional[str] = None
    keywords: List[str] = None


@dataclass
class ConversationTurn:
    """对话轮次数据结构"""
    question: str
    answer: str
    references: List[Dict]
    timestamp: float


class EnhancedAttachment14ManualQA:
    def __init__(self, manual_path: str, max_context_length: int = 32000):
        """
        增强版附件14手册问答系统

        Args:
            manual_path: 手册文件路径
            max_context_length: 最大上下文长度（tokens）
        """
        self.manual_path = manual_path
        self.max_context_length = max_context_length
        self.content = self._load_manual()
        self.structure = self._parse_structure()
        self.chunked_content = self._chunk_content()
        self.conversation_history: List[ConversationTurn] = []
        self.keyword_index = self._build_keyword_index()
        self.session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]

        print(f"✓ 系统初始化完成")
        print(f"✓ 加载章节: {len(self.structure['chapters'])}个")
        print(f"✓ 内容块数: {len(self.chunked_content)}个")
        print(f"✓ 索引关键词: {len(self.keyword_index)}个")

    def _load_manual(self) -> str:
        """加载手册内容"""
        try:
            with open(self.manual_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                print(f"✓ 已加载手册，长度: {len(content)} 字符")
                return content
        except Exception as e:
            print(f"✗ 加载手册失败: {e}")
            # 尝试其他编码
            try:
                with open(self.manual_path, 'r', encoding='gbk', errors='ignore') as f:
                    content = f.read()
                    print(f"✓ 已加载手册(GBK编码)，长度: {len(content)} 字符")
                    return content
            except:
                return ""

    def _parse_structure(self) -> Dict:
        """解析手册结构"""
        structure = {
            "chapters": {},
            "sections": {},
            "definitions": {},
            "tables": {},
            "figures": {},
            "toc": []  # 目录条目
        }

        # 提取所有标题结构
        lines = self.content.split('\n')
        current_chapter = None

        for line in lines:
            line = line.strip()

            # 提取章节
            chapter_match = re.match(r'^##\s*第\s*([一二三四五六七八九十\d]+)\s*章\s*(.+)$', line)
            if chapter_match:
                chapter_num = chapter_match.group(1)
                chapter_title = chapter_match.group(2).strip()
                chapter_key = f"第{chapter_num}章"
                structure["chapters"][chapter_key] = chapter_title
                structure["toc"].append({
                    "level": 2,
                    "type": "chapter",
                    "number": chapter_key,
                    "title": chapter_title
                })
                current_chapter = chapter_key
                continue

            # 提取小节
            section_match = re.match(r'^###\s*(\d+\.\d+(?:\.\d+)*)\s*(.+)$', line)
            if section_match and current_chapter:
                section_num = section_match.group(1)
                section_title = section_match.group(2).strip()
                section_key = f"{current_chapter}_{section_num}"
                structure["sections"][section_key] = {
                    "title": section_title,
                    "chapter": current_chapter,
                    "number": section_num
                }
                structure["toc"].append({
                    "level": 3,
                    "type": "section",
                    "chapter": current_chapter,
                    "number": section_num,
                    "title": section_title
                })
                continue

            # 提取子小节
            subsection_match = re.match(r'^####\s*(\d+\.\d+\.\d+(?:\.\d+)*)\s*(.+)$', line)
            if subsection_match and current_chapter:
                subsection_num = subsection_match.group(1)
                subsection_title = subsection_match.group(2).strip()
                structure["toc"].append({
                    "level": 4,
                    "type": "subsection",
                    "chapter": current_chapter,
                    "number": subsection_num,
                    "title": subsection_title
                })

        # 提取定义（缩写和符号部分）
        def_pattern = r'([A-Za-z][A-Za-z0-9\s\-/]+?)\s*[—–\-]\s*(.+?)(?=\n|$)'
        def_sections = re.finditer(r'##\s*.*?(?:缩写|符号|定义).*?\n(.*?)(?=\n##|$)',
                                   self.content, re.DOTALL | re.IGNORECASE)

        for match in def_sections:
            def_text = match.group(1)
            definitions = re.findall(def_pattern, def_text)
            for key, value in definitions:
                key_clean = key.strip()
                value_clean = value.strip()
                if len(key_clean) > 1 and len(value_clean) > 3:
                    structure["definitions"][key_clean] = value_clean

        # 提取表格和图片引用
        table_pattern = r'表\s*(\d+\.\d+(?:\.\d+)*)[\.\s]*([^。]+)'
        figure_pattern = r'图\s*(\d+\.\d+(?:\.\d+)*)[\.\s]*([^。]+)'

        for match in re.finditer(table_pattern, self.content):
            table_num = match.group(1)
            table_desc = match.group(2).strip()
            structure["tables"][f"表{table_num}"] = table_desc

        for match in re.finditer(figure_pattern, self.content):
            fig_num = match.group(1)
            fig_desc = match.group(2).strip()
            structure["figures"][f"图{fig_num}"] = fig_desc

        return structure

    def _chunk_content(self, chunk_size: int = 2000) -> List[Dict]:
        """将内容分块，便于检索"""
        chunks = []
        lines = self.content.split('\n')

        current_chunk = []
        current_chapter = ""
        current_section = ""

        for line in lines:
            # 检测章节标题
            chapter_match = re.match(r'^##\s*第\s*([一二三四五六七八九十\d]+)\s*章\s*(.+)$', line)
            if chapter_match:
                if current_chunk:
                    chunks.append({
                        "content": '\n'.join(current_chunk),
                        "chapter": current_chapter,
                        "section": current_section,
                        "keywords": self._extract_keywords('\n'.join(current_chunk))
                    })
                    current_chunk = []

                chapter_num = chapter_match.group(1)
                chapter_title = chapter_match.group(2).strip()
                current_chapter = f"第{chapter_num}章 {chapter_title}"
                current_section = ""
                current_chunk.append(line)
                continue

            # 检测小节标题
            section_match = re.match(r'^###\s*(\d+\.\d+(?:\.\d+)*)\s*(.+)$', line)
            if section_match:
                if current_chunk and len('\n'.join(current_chunk)) > 100:
                    chunks.append({
                        "content": '\n'.join(current_chunk),
                        "chapter": current_chapter,
                        "section": current_section,
                        "keywords": self._extract_keywords('\n'.join(current_chunk))
                    })
                    current_chunk = []

                section_num = section_match.group(1)
                section_title = section_match.group(2).strip()
                current_section = f"{section_num} {section_title}"
                current_chunk.append(line)
                continue

            current_chunk.append(line)

            # 如果当前块太大，分割
            if len('\n'.join(current_chunk)) > chunk_size:
                chunks.append({
                    "content": '\n'.join(current_chunk),
                    "chapter": current_chapter,
                    "section": current_section,
                    "keywords": self._extract_keywords('\n'.join(current_chunk))
                })
                current_chunk = []

        # 添加最后一个块
        if current_chunk:
            chunks.append({
                "content": '\n'.join(current_chunk),
                "chapter": current_chapter,
                "section": current_section,
                "keywords": self._extract_keywords('\n'.join(current_chunk))
            })

        return chunks

    def _build_keyword_index(self) -> Dict[str, List[int]]:
        """构建关键词索引"""
        index = defaultdict(list)

        for i, chunk in enumerate(self.chunked_content):
            if "keywords" in chunk:
                for keyword in chunk["keywords"]:
                    index[keyword.lower()].append(i)

        # 添加缩写到索引
        for abbr in self.structure["definitions"].keys():
            index[abbr.lower()] = list(range(len(self.chunked_content)))  # 缩写在所有内容中搜索

        return index

    def _extract_keywords(self, text: str, max_keywords: int = 20) -> List[str]:
        """从文本中提取关键词"""
        keywords = set()

        # 提取大写缩写
        abbreviations = re.findall(r'\b[A-Z]{2,}[A-Z0-9/]*\b', text)
        keywords.update(abbreviations)

        # 提取中文专业术语
        chinese_terms = re.findall(r'[\u4e00-\u9fa5]{2,8}', text)

        # 过滤常见词
        stop_words = {'可以', '一个', '进行', '需要', '要求', '如果', '应当', '必须', '不得'}
        for term in chinese_terms:
            if term not in stop_words and len(term) >= 2:
                # 只保留出现频率较高的术语
                if text.count(term) >= 2:
                    keywords.add(term)

        # 提取数字相关术语
        number_refs = re.findall(r'(?:第[一二三四五六七八九十\d]+章|第\d+\.\d+条|表\d+\.\d+|图\d+\.\d+)', text)
        keywords.update(number_refs)

        # 机场特定术语
        airport_terms = ['跑道', '滑行道', '机坪', '航站楼', '灯光', '标志', '标记', '道面',
                         '净空', '障碍物', 'ILS', 'VOR', 'NDB', 'PCN', 'ACN', 'RESA',
                         '跑道端安全区', '升降带', '精密进近', '非精密进近']

        for term in airport_terms:
            if term in text:
                keywords.add(term)

        return list(keywords)[:max_keywords]

    def get_table_of_contents(self, detailed: bool = True) -> str:
        """获取目录"""
        toc_lines = ["=" * 80]
        toc_lines.append("附件14第I卷（机场设计与运行）目录")
        toc_lines.append("=" * 80)

        for item in self.structure["toc"]:
            indent = "  " * (item["level"] - 2)

            if item["type"] == "chapter":
                toc_lines.append(f"{indent}{item['number']} {item['title']}")
            elif item["type"] == "section":
                toc_lines.append(f"{indent}  {item['number']} {item['title']}")
            elif item["type"] == "subsection" and detailed:
                toc_lines.append(f"{indent}    {item['number']} {item['title']}")

        # 添加定义部分
        if self.structure["definitions"]:
            toc_lines.append("\n缩写和符号表:")
            definitions_list = list(self.structure["definitions"].items())[:15]
            for abbr, meaning in definitions_list:
                toc_lines.append(f"  {abbr} — {meaning}")
            if len(self.structure["definitions"]) > 15:
                toc_lines.append(f"  ... 还有{len(self.structure["definitions"]) - 15}个定义")

        # 添加常用搜索建议
        toc_lines.append("\n" + "-" * 80)
        toc_lines.append("常用搜索关键词:")
        common_keywords = [
            "跑道宽度", "跑道长度", "滑行道", "机坪", "标志", "标记", "灯光",
            "PCN", "ACN", "ILS", "障碍物", "净空", "跑道端安全区", "升降带"
        ]
        toc_lines.append("  " + " | ".join(common_keywords))

        return '\n'.join(toc_lines)

    def semantic_search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """语义搜索相关段落"""
        query_keywords = self._extract_keywords(query, max_keywords=10)

        # 计算相关性分数
        scores = []
        for i, chunk in enumerate(self.chunked_content):
            score = 0

            # 关键词匹配
            chunk_text = chunk["content"].lower()
            for keyword in query_keywords:
                keyword_lower = keyword.lower()
                if keyword_lower in chunk_text:
                    # 计算TF
                    tf = chunk_text.count(keyword_lower) / len(chunk_text.split())
                    score += tf * 10

                    # 标题中的关键词权重更高
                    if chunk.get("section") and keyword_lower in chunk["section"].lower():
                        score += 5
                    if chunk.get("chapter") and keyword_lower in chunk["chapter"].lower():
                        score += 3

            # 考虑章节的重要性
            if chunk.get("chapter") and any(term in chunk["chapter"] for term in ["定义", "术语", "总则"]):
                score *= 0.8  # 降低定义章节的权重

            if score > 0:
                scores.append((i, score))

        # 按分数排序
        scores.sort(key=lambda x: x[1], reverse=True)

        # 构建结果
        results = []
        for idx, score in scores[:top_k]:
            chunk = self.chunked_content[idx]
            # 提取查询相关上下文
            context = self._extract_relevant_context(chunk["content"], query)

            results.append(SearchResult(
                content=context,
                chapter=chunk.get("chapter", ""),
                section=chunk.get("section", ""),
                confidence=min(score / 100, 1.0),
                keywords=self._extract_keywords(context, max_keywords=5)
            ))

        return results

    def _extract_relevant_context(self, text: str, query: str, context_chars: int = 800) -> str:
        """提取最相关的上下文片段"""
        # 找到关键词最密集的区域
        query_keywords = self._extract_keywords(query, max_keywords=10)

        lines = text.split('\n')
        best_start = 0
        best_score = 0

        for i in range(len(lines)):
            window_lines = lines[i:i + 10]
            window_text = '\n'.join(window_lines)

            score = 0
            for keyword in query_keywords:
                if keyword.lower() in window_text.lower():
                    score += 1
                    # 标题中的关键词权重更高
                    if any(line.strip().startswith('#') for line in window_lines):
                        score += 2

            if score > best_score:
                best_score = score
                best_start = i

        # 提取上下文
        start = max(0, best_start - 5)
        end = min(len(lines), best_start + 15)
        context_lines = lines[start:end]

        return '\n'.join(context_lines)

    def generate_search_suggestions(self, question: str) -> List[str]:
        """生成搜索建议"""
        keywords = self._extract_keywords(question, max_keywords=10)
        suggestions = []

        # 基于关键词的章节建议
        keyword_to_chapter = {
            '跑道': ['第3章 物理特性', '第5章 目视助航设施'],
            '滑行道': ['第3章 物理特性', '第5章 目视助航设施'],
            '灯光': ['第5章 目视助航设施'],
            '标志': ['第5章 目视助航设施'],
            '障碍物': ['第4章 障碍物的限制和移除'],
            '净空': ['第4章 障碍物的限制和移除'],
            'PCN': ['第2章 机场数据', '附录1'],
            'ACN': ['第2章 机场数据', '附录1'],
            'ILS': ['第5章 目视助航设施', '附录'],
            '精密进近': ['第5章 目视助航设施', '附录'],
        }

        for keyword in keywords:
            if keyword in keyword_to_chapter:
                suggestions.extend(keyword_to_chapter[keyword])

        # 去重
        suggestions = list(dict.fromkeys(suggestions))

        # 如果没有具体建议，给出一般性建议
        if not suggestions:
            suggestions = [
                "查看第3章 '物理特性' 获取跑道、滑行道尺寸要求",
                "查看第5章 '目视助航设施' 获取灯光、标志规范",
                "查看第4章 '障碍物的限制和移除' 获取净空要求",
                "使用缩写如 'PCN', 'ACN', 'RESA' 进行精确搜索"
            ]

        return suggestions[:5]

    def ask_question(self, question: str, use_ai: bool = True) -> Dict:
        """
        回答问题（支持多轮对话）

        Args:
            question: 用户问题
            use_ai: 是否使用AI生成答案

        Returns:
            包含答案和参考信息的字典
        """
        start_time = time.time()

        print(f"\n🔍 正在搜索: '{question}'")

        # 1. 语义搜索
        search_results = self.semantic_search(question, top_k=5)
        search_time = time.time() - start_time
        print(f"✓ 搜索完成，找到 {len(search_results)} 个相关段落，耗时: {search_time:.2f}秒")

        # 2. 生成搜索建议
        suggestions = self.generate_search_suggestions(question)

        # 3. 准备上下文
        context = self._prepare_context(question, search_results)

        # 4. 生成答案
        if use_ai and search_results:
            print("🤖 正在生成AI答案...")
            answer, confidence = self._generate_ai_answer_with_context(question, context, search_results)
        else:
            print("📝 生成基于检索的答案...")
            answer = self._generate_retrieval_answer(question, search_results)
            confidence = 0.7 if search_results else 0.3

        # 5. 构建响应
        response = {
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "references": [],
            "search_suggestions": suggestions,
            "related_keywords": self._extract_keywords(question, max_keywords=8),
            "search_time": search_time,
            "sources": []
        }

        # 6. 添加引用来源
        for result in search_results:
            if result.confidence > 0.3:  # 只添加置信度较高的来源
                response["references"].append({
                    "content": result.content[:300] + "..." if len(result.content) > 300 else result.content,
                    "chapter": result.chapter,
                    "section": result.section,
                    "confidence": result.confidence,
                    "keywords": result.keywords
                })

                # 添加到源列表
                source_id = f"{result.chapter}_{hash(result.content) % 10000:04d}"
                response["sources"].append({
                    "id": source_id,
                    "ref": f"来自{result.chapter}，{result.section}",
                    "excerpt": result.content[:150] + "..."
                })

        # 7. 记录对话历史
        conversation_turn = ConversationTurn(
            question=question,
            answer=answer,
            references=response["references"],
            timestamp=time.time()
        )
        self.conversation_history.append(conversation_turn)

        # 限制历史长度
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

        return response

    def _prepare_context(self, question: str, search_results: List[SearchResult], max_tokens: int = 30000) -> str:
        """准备上下文信息"""
        context_parts = []

        # 添加会话历史（最近3轮）
        if len(self.conversation_history) > 1:
            context_parts.append("之前的对话:")
            for i, turn in enumerate(self.conversation_history[-3:-1]):
                context_parts.append(f"用户: {turn.question}")
                context_parts.append(f"系统: {turn.answer[:200]}...")
            context_parts.append("")

        # 添加相关搜索结果
        context_parts.append("相关手册内容:")

        used_content = set()
        total_length = 0

        for result in search_results:
            if result.confidence > 0.2:  # 只添加置信度较高的结果
                content_hash = hash(result.content[:500])
                if content_hash not in used_content:
                    content_with_ref = f"[来源: {result.chapter}, {result.section}]\n{result.content}\n"

                    if total_length + len(content_with_ref) < max_tokens:
                        context_parts.append(content_with_ref)
                        total_length += len(content_with_ref)
                        used_content.add(content_hash)

        # 添加相关定义
        context_parts.append("\n相关定义:")
        question_keywords = self._extract_keywords(question)
        for keyword in question_keywords[:5]:
            definition = self.get_definition(keyword)
            if definition:
                context_parts.append(f"{keyword}: {definition}")

        return '\n'.join(context_parts)

    def _generate_ai_answer_with_context(self, question: str, context: str, search_results: List[SearchResult]) -> \
    Tuple[str, float]:
        """使用AI生成答案（带上下文）"""
        try:
            # 构建prompt
            prompt = f"""你是一名国际民航组织附件14（机场设计与运行）的专家。基于以下手册内容回答问题。

{context}

当前问题：{question}

请按以下要求回答：
1. 提供专业、准确的回答，直接针对问题
2. 引用具体来源（章节号、小节号）
3. 如果信息不完整，基于相关知识给出可能的答案，并说明不确定性
4. 回答要具体，避免模糊表述
5. 对于操作性问题，给出具体步骤或标准
6. 如果相关，提及相关表格或图表

专业回答："""

            # 调用API
            response = client.chat.completions.create(
                model="Qwen/Qwen2.5-72B-Instruct",
                messages=[
                    {"role": "system",
                     "content": "你是国际民航组织附件14（机场设计与运行）专家，专门为机场工作人员提供专业指导。用中文回答，保持专业但易懂。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000,
                top_p=0.9
            )

            answer = response.choices[0].message.content

            # 计算置信度（基于搜索结果的平均置信度）
            if search_results:
                avg_confidence = sum(r.confidence for r in search_results) / len(search_results)
            else:
                avg_confidence = 0.5

            # 如果回答中包含"不确定"、"不知道"等词，降低置信度
            uncertainty_indicators = ['不确定', '无法确定', '不知道', '没有找到', '未提及', '请查阅']
            if any(indicator in answer for indicator in uncertainty_indicators):
                avg_confidence *= 0.7

            return answer, min(avg_confidence, 0.95)

        except Exception as e:
            print(f"⚠️ AI生成失败: {e}")
            # 回退到检索答案
            backup_answer = self._generate_retrieval_answer(question, search_results)
            return backup_answer, 0.5

    def _generate_retrieval_answer(self, question: str, search_results: List[SearchResult]) -> str:
        """基于检索结果生成答案"""
        if not search_results:
            return "在手册中没有找到直接相关的内容。建议：\n1. 查看第3章 '物理特性' 和第5章 '目视助航设施'\n2. 尝试使用更具体的术语，如 '跑道宽度' 而非 '跑道'\n3. 查看缩写表获取术语定义"

        # 整理答案
        answer_parts = ["基于附件14手册，相关信息如下：\n"]

        # 添加定义
        question_keywords = self._extract_keywords(question)
        definitions_found = []

        for keyword in question_keywords[:3]:
            definition = self.get_definition(keyword)
            if definition:
                definitions_found.append(f"• {keyword}: {definition}")

        if definitions_found:
            answer_parts.append("相关定义：")
            answer_parts.extend(definitions_found)
            answer_parts.append("")

        # 添加主要内容
        answer_parts.append("手册相关内容：")

        for i, result in enumerate(search_results[:3], 1):
            source_info = f"[来源: {result.chapter}"
            if result.section:
                source_info += f", {result.section}"
            source_info += "]"

            summary = result.content[:200].replace('\n', ' ')
            if len(result.content) > 200:
                summary += "..."

            answer_parts.append(f"{i}. {source_info} {summary}")

        # 添加建议
        answer_parts.append("\n进一步建议：")
        answer_parts.append("• 查看具体章节获取详细信息")
        answer_parts.append("• 注意标准的适用范围和条件")
        answer_parts.append("• 实际应用时请参考最新版本和当地规章")

        return '\n'.join(answer_parts)

    def get_definition(self, term: str) -> Optional[str]:
        """获取术语定义"""
        term_clean = term.strip()

        # 直接查找
        if term_clean in self.structure["definitions"]:
            return self.structure["definitions"][term_clean]

        # 尝试查找近似
        for key, value in self.structure["definitions"].items():
            if term_clean.upper() == key.upper() or term_clean in value:
                return f"{key}: {value}"

        # 尝试部分匹配
        for key, value in self.structure["definitions"].items():
            if term_clean.upper() in key.upper() or key.upper() in term_clean.upper():
                return f"{key}: {value}"

        return None

    def show_conversation_history(self, max_turns: int = 5) -> str:
        """显示对话历史"""
        if not self.conversation_history:
            return "暂无对话历史。"

        history_lines = ["=" * 60]
        history_lines.append("对话历史")
        history_lines.append("=" * 60)

        start_idx = max(0, len(self.conversation_history) - max_turns)

        for i, turn in enumerate(self.conversation_history[start_idx:], start_idx + 1):
            history_lines.append(f"\n[{i}] Q: {turn.question}")
            history_lines.append(f"   A: {turn.answer[:150]}...")
            history_lines.append(f"   时间: {time.strftime('%H:%M:%S', time.localtime(turn.timestamp))}")

        return '\n'.join(history_lines)

    def get_system_status(self) -> Dict:
        """获取系统状态"""
        return {
            "session_id": self.session_id,
            "manual_loaded": len(self.content) > 0,
            "content_length": len(self.content),
            "chapters_count": len(self.structure["chapters"]),
            "definitions_count": len(self.structure["definitions"]),
            "conversation_turns": len(self.conversation_history),
            "keyword_index_size": len(self.keyword_index),
            "chunks_count": len(self.chunked_content),
            "max_context_length": self.max_context_length
        }


def display_answer(response: Dict):
    """美观地显示答案"""
    print("\n" + "=" * 80)
    print("📋 问题:", response["question"])
    print("=" * 80)

    # 显示置信度
    confidence_emoji = "🔴"
    if response["confidence"] > 0.8:
        confidence_emoji = "🟢"
    elif response["confidence"] > 0.6:
        confidence_emoji = "🟡"

    print(f"{confidence_emoji} 置信度: {response['confidence']:.1%}")
    print(f"⏱️ 搜索耗时: {response['search_time']:.2f}秒")

    print("\n" + "-" * 80)
    print("💡 答案:")
    print("-" * 80)
    print(response["answer"])

    # 显示来源
    if response.get("references"):
        print("\n" + "-" * 80)
        print("📚 参考来源:")
        print("-" * 80)
        for i, ref in enumerate(response["references"][:3], 1):
            print(f"\n{i}. {ref['chapter']}")
            if ref.get('section'):
                print(f"   小节: {ref['section']}")
            print(f"   相关度: {ref['confidence']:.1%}")
            print(f"   内容摘要: {ref['content'][:200]}...")

    # 显示关键词
    if response.get("related_keywords"):
        print("\n" + "-" * 80)
        print("🔑 相关关键词:")
        print("-" * 80)
        print(" | ".join(response["related_keywords"][:10]))

    # 显示建议
    if response.get("search_suggestions"):
        print("\n" + "-" * 80)
        print("💡 搜索建议:")
        print("-" * 80)
        for i, suggestion in enumerate(response["search_suggestions"], 1):
            print(f"{i}. {suggestion}")

    print("\n" + "=" * 80)


def main():
    """主函数"""
    # 设置文件路径
    manual_path = r"D:\AlgorithmClub\Damoxingyuanli\homework\datas\附件14 机场  — 机场设计与运行_第I卷 (第九版，2022年7月)\index.md"

    print("🚀 正在初始化附件14手册问答系统...")
    print(f"📂 文件路径: {manual_path}")

    try:
        # 初始化系统
        qa_system = EnhancedAttachment14ManualQA(manual_path)

        # 显示系统状态
        status = qa_system.get_system_status()
        print(f"\n✅ 系统状态:")
        print(f"   • 会话ID: {status['session_id']}")
        print(f"   • 手册加载: {'成功' if status['manual_loaded'] else '失败'}")
        print(f"   • 章节数: {status['chapters_count']}")
        print(f"   • 定义数: {status['definitions_count']}")
        print(f"   • 内容块: {status['chunks_count']}")

        # 显示目录
        print("\n" + "=" * 80)
        toc = qa_system.get_table_of_contents(detailed=True)
        print(toc)

        # 示例问题
        print("\n" + "=" * 80)
        print("💡 示例问题（您可以直接输入数字选择）:")
        print("=" * 80)
        example_questions = [
            "1. 跑道端安全区(RESA)的尺寸要求是什么？",
            "2. PCN和ACN分别代表什么？如何计算？",
            "3. 跑道宽度和长度的基本要求是什么？",
            "4. 目视进近坡度指示系统(VASIS)的布置要求？",
            "5. 障碍物限制面包括哪些？各自的标准是什么？",
            "6. 滑行道的最小宽度要求是多少？",
            "7. 跑道标志和滑行道标志有什么区别？",
            "8. 机场灯光系统有哪些类型？",
            "9. 精密进近跑道和非精密进近跑道的区别？",
            "10. 机场道面强度报告PCN如何解读？"
        ]

        for q in example_questions:
            print(q)

        print("\n" + "=" * 80)
        print("💬 开始对话 (输入 'help' 查看帮助, 'quit' 退出)")
        print("=" * 80)

        # 交互式问答
        while True:
            try:
                print("\n" + "-" * 80)
                user_input = input("\n💭 请输入问题或命令: ").strip()

                if not user_input:
                    continue

                # 命令处理
                if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                    print("👋 感谢使用，再见！")
                    break

                elif user_input.lower() in ['help', '帮助', '?']:
                    print("\n📋 可用命令:")
                    print("  help      - 显示帮助")
                    print("  toc       - 显示目录")
                    print("  history   - 显示对话历史")
                    print("  status    - 显示系统状态")
                    print("  keywords  - 显示常用关键词")
                    print("  clear     - 清除对话历史")
                    print("  quit      - 退出系统")
                    print("\n💡 提示：")
                    print("  • 输入数字1-10选择示例问题")
                    print("  • 使用具体术语提问更精确")
                    print("  • 可以连续提问，系统会记住上下文")
                    continue

                elif user_input.lower() == 'toc':
                    print(qa_system.get_table_of_contents(detailed=True))
                    continue

                elif user_input.lower() == 'history':
                    print(qa_system.show_conversation_history())
                    continue

                elif user_input.lower() == 'status':
                    status = qa_system.get_system_status()
                    print(f"\n📊 系统状态:")
                    for key, value in status.items():
                        print(f"  {key}: {value}")
                    continue

                elif user_input.lower() == 'keywords':
                    print("\n🔑 常用关键词:")
                    keywords = [
                        "跑道", "滑行道", "机坪", "灯光", "标志", "PCN", "ACN",
                        "ILS", "VOR", "NDB", "RESA", "障碍物", "净空", "道面",
                        "升降带", "跑道端", "精密进近", "非精密进近", "目视助航"
                    ]
                    print(" | ".join(keywords))
                    continue

                elif user_input.lower() == 'clear':
                    qa_system.conversation_history = []
                    print("🗑️ 对话历史已清除")
                    continue

                # 处理数字选择示例问题
                elif user_input.isdigit() and 1 <= int(user_input) <= len(example_questions):
                    idx = int(user_input) - 1
                    actual_question = example_questions[idx].split('. ', 1)[1]
                    print(f"\n📝 选择问题: {actual_question}")
                    user_input = actual_question

                # 处理问题
                print(f"\n🧠 正在分析: '{user_input}'")

                response = qa_system.ask_question(user_input, use_ai=True)
                display_answer(response)

            except KeyboardInterrupt:
                print("\n\n⚠️ 中断操作，输入 'quit' 退出")
                continue
            except Exception as e:
                print(f"\n❌ 发生错误: {e}")
                print("请重新输入或输入 'quit' 退出")
                continue

    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 '{manual_path}'")
        print("请检查文件路径是否正确")
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        print("请检查: 1) 文件路径 2) API密钥 3) 网络连接")


if __name__ == "__main__":
    main()
