import os
import json
import re
from pathlib import Path
import hashlib
import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# 设置硅基流动API
import openai

openai.api_base = "https://api.siliconflow.cn/v1"
openai.api_key = "sk-bdgrimfksplnwstzulxfsrdijhjqribunforxvknatzpjlui"


class Attachment14ManualQA:
    def __init__(self, manual_path: str):
        """
        初始化附件14手册问答系统

        Args:
            manual_path: 手册文件路径
        """
        self.manual_path = manual_path
        self.content = self._load_manual()
        self.structure = self._parse_structure()
        self.index = self._build_index()

    def _load_manual(self) -> str:
        """加载手册内容"""
        try:
            with open(self.manual_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"加载手册失败: {e}")
            return ""

    def _parse_structure(self) -> Dict:
        """解析手册结构"""
        structure = {
            "chapters": {},
            "sections": {},
            "definitions": {},
            "tables": {},
            "figures": {}
        }

        # 使用正则表达式提取章节
        chapter_pattern = r'## 第(\d+)章([^#\n]+)'
        section_pattern = r'### (\d+\.\d+[^#\n]+)'
        definition_pattern = r'([A-Za-z]+[^—\n]{0,50})—([^\n]+)'

        # 提取章节
        chapters = re.findall(chapter_pattern, self.content)
        for num, title in chapters:
            structure["chapters"][f"第{num}章"] = title.strip()

        # 提取小节
        sections = re.findall(section_pattern, self.content)
        for section in sections:
            if '.' in section:
                parts = section.split(' ', 1)
                if len(parts) > 1:
                    structure["sections"][parts[0]] = parts[1].strip()

        # 提取定义（缩写和符号部分）
        def_section = re.search(r'## 缩写和符号[^#]+', self.content, re.DOTALL)
        if def_section:
            def_text = def_section.group(0)
            lines = def_text.split('\n')
            for i, line in enumerate(lines):
                if '—' in line and len(line.strip()) < 100:  # 简单的定义行检测
                    parts = line.split('—', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        structure["definitions"][key] = value

        return structure

    def _build_index(self) -> Dict:
        """构建内容索引"""
        index = {
            "by_chapter": {},
            "by_keyword": {},
            "by_abbreviation": {}
        }

        # 按章节索引
        chapter_pattern = r'(## 第\d+章[^#]+)'
        chapters = re.split(chapter_pattern, self.content)[1:]

        for i in range(0, len(chapters), 2):
            if i + 1 < len(chapters):
                chapter_title = chapters[i].strip()
                chapter_content = chapters[i + 1]
                index["by_chapter"][chapter_title] = chapter_content

                # 提取关键词
                words = re.findall(r'[A-Z]{2,}[A-Z0-9]*|[A-Z]{2,}|[\u4e00-\u9fa5]{2,6}',
                                   chapter_title + chapter_content)
                for word in set(words):
                    if word not in index["by_keyword"]:
                        index["by_keyword"][word] = []
                    index["by_keyword"][word].append({
                        "chapter": chapter_title,
                        "content": chapter_content[:500] + "..."
                    })

        return index

    def search_by_keyword(self, keyword: str, limit: int = 5) -> List[Dict]:
        """关键词搜索"""
        results = []
        keyword_lower = keyword.lower()

        # 在缩写中搜索
        for abbr, meaning in self.structure["definitions"].items():
            if keyword_lower in abbr.lower() or keyword_lower in meaning.lower():
                results.append({
                    "type": "definition",
                    "key": abbr,
                    "value": meaning,
                    "source": "缩写和符号表"
                })

        # 在章节标题中搜索
        for chapter_title in self.structure["chapters"].values():
            if keyword_lower in chapter_title.lower():
                chapter_num = [k for k, v in self.structure["chapters"].items()
                               if v == chapter_title][0]
                results.append({
                    "type": "chapter",
                    "title": f"{chapter_num} {chapter_title}",
                    "source": "章节目录"
                })

        # 在内容中搜索
        for chapter_title, chapter_content in self.index["by_chapter"].items():
            if keyword_lower in chapter_content.lower():
                # 提取相关上下文
                start_idx = max(0, chapter_content.lower().find(keyword_lower) - 100)
                end_idx = min(len(chapter_content),
                              chapter_content.lower().find(keyword_lower) + 300)
                context = chapter_content[start_idx:end_idx]

                results.append({
                    "type": "content",
                    "chapter": chapter_title,
                    "context": f"...{context}...",
                    "source": chapter_title
                })

        return results[:limit]

    def get_definition(self, term: str) -> Optional[str]:
        """获取术语定义"""
        # 直接查找
        if term in self.structure["definitions"]:
            return self.structure["definitions"][term]

        # 尝试查找近似
        for key, value in self.structure["definitions"].items():
            if term.upper() in key.upper() or term in value:
                return f"{key}: {value}"

        return None

    def ask_question(self, question: str, use_ai: bool = True) -> Dict:
        """
        回答问题

        Args:
            question: 用户问题
            use_ai: 是否使用AI生成答案

        Returns:
            包含答案和参考信息的字典
        """
        response = {
            "question": question,
            "answer": "",
            "references": [],
            "suggestions": []
        }

        # 1. 提取问题中的关键词
        keywords = self._extract_keywords(question)
        response["keywords"] = keywords

        # 2. 搜索相关信息
        search_results = []
        for keyword in keywords:
            results = self.search_by_keyword(keyword, limit=3)
            search_results.extend(results)

        # 3. 去重
        unique_results = []
        seen = set()
        for result in search_results:
            key = str(result.get("key", "")) + str(result.get("chapter", "")) + str(result.get("context", ""))
            if key not in seen:
                unique_results.append(result)
                seen.add(key)

        response["references"] = unique_results

        # 4. 生成答案
        if use_ai:
            answer = self._generate_ai_answer(question, unique_results)
            response["answer"] = answer
        else:
            # 基于搜索结果的简单答案
            if unique_results:
                response["answer"] = self._generate_simple_answer(question, unique_results)
            else:
                response["answer"] = "未找到相关信息。请尝试使用不同的关键词或查看手册目录。"

        # 5. 添加建议
        response["suggestions"] = self._generate_suggestions(question, keywords)

        return response

    def _extract_keywords(self, text: str) -> List[str]:
        """从文本中提取关键词"""
        # 提取大写缩写
        abbreviations = re.findall(r'\b[A-Z]{2,}[A-Z0-9]*\b', text)

        # 提取中文术语
        chinese_terms = re.findall(r'[\u4e00-\u9fa5]{2,6}', text)

        # 提取数字相关的术语（如第X章）
        chapter_refs = re.findall(r'第[一二三四五六七八九十\d]+章', text)

        keywords = abbreviations + chinese_terms + chapter_refs

        # 去重并过滤
        unique_keywords = []
        seen = set()
        for kw in keywords:
            if kw not in seen and len(kw) > 1:
                unique_keywords.append(kw)
                seen.add(kw)

        return unique_keywords

    def _generate_simple_answer(self, question: str, references: List[Dict]) -> str:
        """基于搜索结果生成简单答案"""
        answer_parts = []

        # 检查是否有定义
        definitions = [r for r in references if r.get("type") == "definition"]
        if definitions:
            answer_parts.append("相关定义：")
            for d in definitions[:3]:
                answer_parts.append(f"  • {d['key']}: {d['value']}")

        # 检查是否有章节内容
        chapters = [r for r in references if r.get("type") in ["chapter", "content"]]
        if chapters:
            answer_parts.append("\n相关内容：")
            for c in chapters[:3]:
                if c.get("type") == "chapter":
                    answer_parts.append(f"  • {c['title']}")
                else:
                    answer_parts.append(f"  • {c['chapter']}")
                    if c.get("context"):
                        answer_parts.append(f"    上下文: {c['context'][:100]}...")

        if not answer_parts:
            return "未找到相关信息。"

        return "\n".join(answer_parts)

    def _generate_ai_answer(self, question: str, references: List[Dict]) -> str:
        """使用AI生成答案"""
        try:
            # 准备上下文
            context_parts = ["国际民航组织附件14第I卷（机场设计与运行）相关信息："]

            # 添加定义
            defs = [r for r in references if r.get("type") == "definition"]
            if defs:
                context_parts.append("\n相关定义：")
                for d in defs[:5]:
                    context_parts.append(f"- {d['key']}: {d['value']}")

            # 添加章节内容
            chapters = [r for r in references if r.get("type") in ["chapter", "content"]]
            if chapters:
                context_parts.append("\n相关章节内容：")
                for c in chapters[:3]:
                    if c.get("type") == "chapter":
                        context_parts.append(f"- 章节: {c.get('title', '')}")
                    else:
                        context_parts.append(f"- 来源: {c.get('chapter', '')}")
                        if c.get("context"):
                            context_parts.append(f"  内容: {c['context']}")

            context = "\n".join(context_parts)

            # 构建提示
            prompt = f"""你是一名国际民航组织附件14的专家。基于以下手册内容，回答用户的问题。

{context}

用户问题：{question}

请提供准确、专业的回答，引用相关章节和定义。如果信息不足，请说明需要查阅手册的具体部分。

回答："""

            # 调用AI API
            response = openai.ChatCompletion.create(
                model="Qwen/Qwen2.5-72B-Instruct",
                messages=[
                    {"role": "system", "content": "你是国际民航组织附件14（机场设计与运行）专家，请用中文回答专业问题。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"AI生成答案时出错: {e}\n\n备用答案:\n{self._generate_simple_answer(question, references)}"

    def _generate_suggestions(self, question: str, keywords: List[str]) -> List[str]:
        """生成搜索建议"""
        suggestions = []

        # 如果有关键词但没有找到结果
        if keywords and len(keywords) > 0:
            suggestions.append(f"尝试搜索关键词: {', '.join(keywords[:3])}")

        # 建议查看相关章节
        if any('跑道' in kw for kw in keywords):
            suggestions.append("查看第3章 '物理特性' 和第5章 '目视助航设施'")
        if any('道面' in kw for kw in keywords):
            suggestions.append("查看第2章 '机场数据' 和第3章 '物理特性'")
        if any('灯光' in kw for kw in keywords):
            suggestions.append("查看第5章 '目视助航设施'")
        if any('障碍' in kw for kw in keywords):
            suggestions.append("查看第4章 '障碍物的限制和移除'")

        # 通用建议
        if not suggestions:
            suggestions = [
                "查看手册目录了解各章内容",
                "使用英文缩写或中文全称进行搜索",
                "尝试更具体的术语，如 '跑道端安全区' 而不是 '跑道'"
            ]

        return suggestions[:3]

    def get_chapter_content(self, chapter_ref: str) -> Optional[str]:
        """获取章节内容"""
        # 尝试匹配章节号
        chapter_patterns = [
            r'第(\d+)章',
            r'第([一二三四五六七八九十]+)章',
            r'chapter\s*(\d+)',
            r'(\d+)\s*章'
        ]

        chapter_num = None
        for pattern in chapter_patterns:
            match = re.search(pattern, chapter_ref, re.IGNORECASE)
            if match:
                if pattern == r'第([一二三四五六七八九十]+)章':
                    # 中文数字转阿拉伯数字
                    cn_num = match.group(1)
                    num_map = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
                               '十': 10}
                    chapter_num = str(num_map.get(cn_num, cn_num))
                else:
                    chapter_num = match.group(1)
                break

        if chapter_num:
            for title, content in self.index["by_chapter"].items():
                if f"第{chapter_num}章" in title:
                    return content

        return None

    def get_manual_structure(self) -> Dict:
        """获取手册结构"""
        return {
            "total_chapters": len(self.structure["chapters"]),
            "chapters": self.structure["chapters"],
            "total_definitions": len(self.structure["definitions"]),
            "sample_definitions": dict(list(self.structure["definitions"].items())[:10])
        }


# 使用示例
def main():
    # 初始化问答系统
    manual_path = r"D:\AlgorithmClub\Damoxingyuanli\homework\datas\附件14 机场  — 机场设计与运行_第I卷 (第九版，2022年7月)\index.md"

    print("正在加载附件14手册...")
    qa_system = Attachment14ManualQA(manual_path)

    # 查看手册结构
    structure = qa_system.get_manual_structure()
    print(f"手册包含 {structure['total_chapters']} 章")
    print(f"手册包含 {structure['total_definitions']} 个定义")
    print("\n章节目录:")
    for num, title in structure["chapters"].items():
        print(f"  {num}: {title}")

    print("\n" + "=" * 80)
    print("附件14第I卷（机场设计与运行）问答系统")
    print("=" * 80)

    # 示例问题
    example_questions = [
        "什么是跑道端安全区？",
        "PCN是什么意思？",
        "跑道宽度有哪些要求？",
        "简述目视进近坡度指示系统",
        "障碍物限制面包括哪些？"
    ]

    print("\n示例问题:")
    for i, q in enumerate(example_questions, 1):
        print(f"{i}. {q}")

    # 交互式问答
    while True:
        print("\n" + "-" * 80)
        user_question = input("\n请输入您的问题（输入'quit'退出）: ").strip()

        if user_question.lower() in ['quit', 'exit', '退出']:
            print("感谢使用，再见！")
            break

        if not user_question:
            continue

        print(f"\n问题: {user_question}")
        print("正在搜索...")

        # 获取答案
        response = qa_system.ask_question(user_question, use_ai=True)

        print("\n" + "=" * 80)
        print("答案:")
        print("=" * 80)
        print(response["answer"])

        if response.get("references"):
            print("\n" + "-" * 80)
            print("参考信息:")
            print("-" * 80)
            for i, ref in enumerate(response["references"][:3], 1):
                if ref.get("type") == "definition":
                    print(f"{i}. 定义: {ref['key']} = {ref['value']}")
                elif ref.get("type") == "chapter":
                    print(f"{i}. 章节: {ref['title']}")
                elif ref.get("type") == "content":
                    print(f"{i}. 内容来自: {ref['chapter']}")
                    if ref.get("context"):
                        print(f"   上下文: {ref['context'][:150]}...")

        if response.get("suggestions"):
            print("\n" + "-" * 80)
            print("搜索建议:")
            print("-" * 80)
            for suggestion in response["suggestions"]:
                print(f"• {suggestion}")


if __name__ == "__main__":
    main()