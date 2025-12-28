"""
智能问答系统 v1.0
作者：AI助手
功能：基于本地知识库和网络搜索的智能问答系统
"""

import os
import json
import requests
import sys
from typing import List, Dict, Tuple, Optional
import numpy as np
from datetime import datetime
import hashlib
import time

# 安装必要依赖：
# pip install chromadb langchain sentence-transformers requests beautifulsoup4 markdown tiktoken

try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
    import markdown
    from bs4 import BeautifulSoup
    import re
    import tiktoken
except ImportError as e:
    print(f"缺少必要的依赖库: {e}")
    print("请运行: pip install chromadb sentence-transformers requests beautifulsoup4 markdown tiktoken")
    sys.exit(1)


class LocalKnowledgeBase:
    """本地知识库管理"""

    def __init__(self, knowledge_base_path: str):
        """
        初始化本地知识库

        Args:
            knowledge_base_path: .md文件或目录路径
        """
        self.knowledge_base_path = knowledge_base_path
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

        # 初始化Chroma向量数据库
        chroma_persist_dir = "D:\\chroma_db"  # 持久化存储目录

        try:
            self.chroma_client = chromadb.PersistentClient(
                path=chroma_persist_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # 创建或获取集合
            self.collection = self.chroma_client.get_or_create_collection(
                name="airport_knowledge_base",
                metadata={"hnsw:space": "cosine"}
            )

            print(f"✅ Chroma数据库初始化成功，存储路径: {chroma_persist_dir}")

        except Exception as e:
            print(f"❌ Chroma数据库初始化失败: {e}")
            print("尝试创建内存数据库...")
            # 回退到内存数据库
            self.chroma_client = chromadb.EphemeralClient()
            self.collection = self.chroma_client.create_collection(
                name="airport_knowledge_base",
                metadata={"hnsw:space": "cosine"}
            )

        # 加载知识库
        self.load_knowledge_base()

    def read_markdown_file(self, filepath: str) -> str:
        """读取Markdown文件内容"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return content
        except Exception as e:
            print(f"读取文件 {filepath} 时出错: {str(e)}")
            return ""

    def split_document(self, content: str, filename: str, max_chunk_size: int = 800) -> List[str]:
        """智能分割文档为片段"""
        chunks = []

        # 1. 首先尝试按标题分割
        # 匹配 #, ##, ### 等标题
        heading_pattern = r'(?m)^(#{1,3})\s+(.+?)$'
        sections = re.split(heading_pattern, content)

        current_chunk = ""
        current_section = ""

        if len(sections) > 1:
            # 有标题的情况
            for i in range(1, len(sections), 3):
                if i + 2 < len(sections):
                    heading_level = sections[i]
                    heading_text = sections[i + 1]
                    section_content = sections[i + 2] if i + 2 < len(sections) else ""

                    # 创建包含标题的块
                    chunk = f"{heading_level} {heading_text}\n{section_content}"

                    # 如果块太大，进一步分割
                    if len(chunk) > max_chunk_size:
                        sub_chunks = self.split_by_paragraphs(chunk, max_chunk_size)
                        chunks.extend(sub_chunks)
                    else:
                        chunks.append(chunk.strip())
        else:
            # 没有标题，按段落分割
            chunks = self.split_by_paragraphs(content, max_chunk_size)

        return chunks

    def split_by_paragraphs(self, text: str, max_chunk_size: int) -> List[str]:
        """按段落分割文本"""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) + 2 <= max_chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = para

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def load_knowledge_base(self):
        """加载知识库到向量数据库"""
        print("🔄 正在加载知识库...")

        # 检查是否已经有数据
        try:
            count = self.collection.count()
            if count > 0:
                print(f"📚 知识库已存在，包含 {count} 个文档片段")
                return
        except:
            pass

        documents = []
        metadatas = []
        ids = []

        # 检查路径是否存在
        if not os.path.exists(self.knowledge_base_path):
            print(f"❌ 警告：知识库路径不存在: {self.knowledge_base_path}")
            print("请检查路径是否正确")
            return

        # 如果是单个文件
        if os.path.isfile(self.knowledge_base_path) and self.knowledge_base_path.endswith('.md'):
            print(f"📄 加载单个文件: {self.knowledge_base_path}")
            files = [self.knowledge_base_path]
        # 如果是目录
        elif os.path.isdir(self.knowledge_base_path):
            print(f"📁 加载目录: {self.knowledge_base_path}")
            files = [os.path.join(self.knowledge_base_path, f) for f in os.listdir(self.knowledge_base_path)
                     if f.endswith('.md')]
        else:
            print(f"❌ 错误：路径既不是文件也不是目录: {self.knowledge_base_path}")
            return

        if not files:
            print("❌ 未找到任何.md文件")
            return

        total_chunks = 0

        for filepath in files:
            try:
                filename = os.path.basename(filepath)
                print(f"  正在处理: {filename}")

                content = self.read_markdown_file(filepath)
                if not content:
                    continue

                # 分割文档
                chunks = self.split_document(content, filename)

                for i, chunk in enumerate(chunks):
                    chunk_id = f"{filename}_{i}_{hashlib.md5(chunk.encode()).hexdigest()[:8]}"
                    documents.append(chunk)
                    metadatas.append({
                        "source": filename,
                        "filepath": filepath,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "timestamp": datetime.now().isoformat()
                    })
                    ids.append(chunk_id)
                    total_chunks += 1

                    if total_chunks % 50 == 0:
                        print(f"  已处理 {total_chunks} 个文档片段...")

            except Exception as e:
                print(f"  处理文件 {filepath} 时出错: {str(e)}")
                continue

        if documents:
            print(f"📊 正在为 {len(documents)} 个文档片段生成嵌入向量...")

            try:
                # 分批处理，避免内存不足
                batch_size = 100
                all_embeddings = []

                for i in range(0, len(documents), batch_size):
                    batch_docs = documents[i:i + batch_size]
                    print(f"  处理批次 {i // batch_size + 1}/{(len(documents) - 1) // batch_size + 1}")

                    batch_embeddings = self.embedding_model.encode(batch_docs).tolist()
                    all_embeddings.extend(batch_embeddings)

                print("✅ 嵌入向量生成完成，正在添加到数据库...")

                # 添加到向量数据库
                self.collection.add(
                    embeddings=all_embeddings,
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )

                print(f"✅ 知识库加载完成，共 {len(documents)} 个文档片段")

            except Exception as e:
                print(f"❌ 添加文档到数据库时出错: {str(e)}")
        else:
            print("❌ 没有找到可加载的文档内容")

    def search(self, query: str, top_k: int = 5, similarity_threshold: float = 0.3) -> List[Dict]:
        """
        在知识库中搜索相关问题

        Args:
            query: 查询文本
            top_k: 返回结果数量
            similarity_threshold: 相似度阈值

        Returns:
            相关文档列表
        """
        try:
            # 生成查询嵌入
            query_embedding = self.embedding_model.encode([query]).tolist()

            # 搜索向量数据库
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

            # 处理结果
            relevant_docs = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                        results['documents'][0],
                        results['metadatas'][0],
                        results['distances'][0]
                )):
                    # 将距离转换为相似度分数（余弦相似度）
                    similarity = 1 - distance

                    if similarity >= similarity_threshold:
                        relevant_docs.append({
                            "content": doc,
                            "metadata": metadata,
                            "similarity": similarity,
                            "source": "local_knowledge_base"
                        })

            return relevant_docs

        except Exception as e:
            print(f"❌ 搜索知识库时出错: {str(e)}")
            return []


class WebSearch:
    """网络搜索模块"""

    def __init__(self, silicon_flow_api_key: str):
        """
        初始化网络搜索模块

        Args:
            silicon_flow_api_key: 硅基流动API密钥
        """
        self.api_key = silicon_flow_api_key
        self.api_url = "https://api.siliconflow.cn/v1/chat/completions"

    def search_web(self, query: str) -> Optional[Dict]:
        """
        使用硅基流动API进行网络搜索
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # 构造请求消息
        messages = [
            {
                "role": "system",
                "content": """你是一个专业的信息助手。请根据用户的查询提供准确、有用的信息。
                如果问题涉及专业领域，请确保信息的准确性。
                如果无法找到确切答案，请提供相关信息和进一步查询的建议。"""
            },
            {
                "role": "user",
                "content": f"请提供关于以下问题的详细、准确的信息：{query}\n请确保信息的准确性和实用性。"
            }
        ]

        try:
            payload = {
                "model": "Qwen/Qwen2.5-72B-Instruct",  # 使用Qwen模型
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 1500,
                "stream": False
            }

            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)

            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']

                return {
                    "content": content,
                    "source": "web_search",
                    "confidence": 0.8,
                    "model": "Qwen2.5-72B"
                }
            else:
                print(f"❌ API请求失败: {response.status_code}")
                print(f"响应内容: {response.text}")
                return None

        except requests.exceptions.Timeout:
            print("❌ 网络搜索请求超时")
            return None
        except Exception as e:
            print(f"❌ 网络搜索时出错: {str(e)}")
            return None


class SmartQASystem:
    """智能问答系统"""

    def __init__(self, knowledge_base_path: str, silicon_flow_api_key: str):
        """
        初始化智能问答系统

        Args:
            knowledge_base_path: 知识库路径
            silicon_flow_api_key: 硅基流动API密钥
        """
        print("=" * 60)
        print("🚀 智能问答系统初始化中...")
        print("=" * 60)

        # 初始化知识库
        print("🔧 正在初始化知识库...")
        self.knowledge_base = LocalKnowledgeBase(knowledge_base_path)

        # 初始化网络搜索
        print("🔧 正在初始化网络搜索模块...")
        self.web_search = WebSearch(silicon_flow_api_key)

        # 初始化会话状态
        self.conversation_history = []
        self.fallback_count = 0
        self.max_fallback_before_escalation = 3
        self.user_name = "用户"

        print("=" * 60)
        print("✅ 智能问答系统初始化完成！")
        print(f"📁 知识库路径: {knowledge_base_path}")
        print(f"🔑 API密钥: {silicon_flow_api_key[:12]}...{silicon_flow_api_key[-8:]}")
        print("=" * 60)
        print("\n💬 请输入您的问题开始对话（输入'退出'或'quit'结束）\n")

    def classify_question(self, question: str) -> str:
        """分类问题类型"""
        question_lower = question.lower()

        # 机场/直升机相关关键词
        airport_keywords = ['机场', '跑道', '航站楼', '塔台', '停机坪', '安检', '海关']
        helicopter_keywords = ['直升机', '旋翼', '起降', '停机坪', '旋翼机']
        aviation_keywords = ['航空', '飞行', '飞行员', '空管', '导航', '仪表']

        # 问题类型检测
        if any(kw in question_lower for kw in ['怎么', '如何', '怎样', '步骤', '方法', '操作']):
            return "how_to"
        elif any(kw in question_lower for kw in ['是什么', '什么是', '定义', '概念', '解释']):
            return "definition"
        elif any(kw in question_lower for kw in ['为什么', '原因', '为何', '原理']):
            return "explanation"
        elif any(kw in question_lower for kw in ['区别', '比较', '对比', '差异', '不同']):
            return "comparison"
        elif any(kw in question_lower for kw in ['谁', '何时', '哪里', '多少', '哪些', '是否']):
            return "factual"
        elif any(kw in question_lower for kw in airport_keywords + helicopter_keywords + aviation_keywords):
            return "aviation_specific"
        else:
            return "general"

    def extract_keywords(self, question: str) -> List[str]:
        """从问题中提取关键词"""
        # 去除停用词
        stop_words = {'的', '了', '和', '是', '就', '都', '而', '及', '与', '或', '在', '有', '更', '这个', '一个',
                      '吗', '呢', '啊', '呀'}

        # 提取中文和英文单词
        words = re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', question)
        keywords = [word for word in words if word not in stop_words and len(word) > 1]

        # 保留前5个关键词
        return keywords[:5]

    def get_available_topics(self) -> List[str]:
        """获取知识库中可用的主题"""
        try:
            # 从向量数据库获取所有文档的元数据
            all_docs = self.knowledge_base.collection.get()

            topics_set = set()
            if all_docs and 'metadatas' in all_docs:
                for metadata in all_docs['metadatas']:
                    if isinstance(metadata, dict):
                        source = metadata.get('source', '')
                        if source and source.endswith('.md'):
                            # 去除.md后缀，获取主题名
                            topic = os.path.splitext(source)[0]
                            topics_set.add(topic)

            topics = list(topics_set)
            return sorted(topics)[:10]  # 返回前10个排序后的主题

        except Exception as e:
            print(f"获取主题列表时出错: {str(e)}")

        # 默认主题（如果获取失败）
        return ["机场设计", "直升机机场", "飞行规则", "安全标准", "运行管理"]

    def format_answer(self, doc: Dict, is_multiple: bool = False) -> str:
        """格式化单个文档的回答"""
        content = doc['content']
        similarity = doc['similarity']
        source = doc['metadata'].get('source', '未知文档')

        # 限制内容长度
        if len(content) > 800:
            content = content[:800] + "...\n\n（内容已截断，完整信息请查看原文档）"

        response = ""
        if not is_multiple:
            response += f"📚 **来自《{source}》** (相关度: {similarity:.1%})\n\n"

        response += f"{content}\n"

        if not is_multiple:
            response += f"\n---\n*来源: {source}*"

        return response

    def format_partial_answer(self, relevant_docs: List[Dict], question: str) -> str:
        """格式化部分相关的答案"""
        if not relevant_docs:
            return self.generate_smart_options(question)

        response = "🤔 我找到了一些相关信息，但可能不完全匹配您的问题：\n\n"

        for i, doc in enumerate(relevant_docs[:3], 1):
            content_preview = doc['content'][:300] + "..." if len(doc['content']) > 300 else doc['content']
            source = doc['metadata'].get('source', '未知文档')
            similarity = doc['similarity']

            response += f"**选项 {i}** - 来自《{source}》 (相关度: {similarity:.1%})\n"
            response += f"{content_preview}\n\n"

        response += "这些信息对您有帮助吗？或者您可以：\n"
        response += "1️⃣ 选择其中一个选项查看详情\n"
        response += "2️⃣ **启用网络搜索**获取更广泛信息\n"
        response += "3️⃣ **重新表述**您的问题\n"
        response += "4️⃣ **查看其他相关主题**\n\n"
        response += "请回复数字选择相应操作。"

        return response

    def generate_smart_options(self, question: str) -> str:
        """生成智能选项菜单"""
        keywords = self.extract_keywords(question)
        available_topics = self.get_available_topics()
        question_type = self.classify_question(question)

        response = "🔍 我未能在知识库中找到确切答案。您可以选择以下操作：\n\n"

        # 基础选项
        options = [
            "1️⃣ 📝 **重新表述问题**（当前表述可能不够明确）",
            "2️⃣ 🔍 **降低搜索标准**（使用更宽松的匹配条件）",
            "3️⃣ 🌐 **启用网络搜索**（获取最新、更广泛的信息）",
            f"4️⃣ 📂 **浏览知识库主题**（当前包含：{', '.join(available_topics[:3])}等）",
        ]

        # 根据关键词添加选项
        if keywords:
            options.append(f"5️⃣ 🔑 **使用关键词搜索**：{', '.join(keywords)}")

        # 根据问题类型添加特定选项
        if question_type == "aviation_specific":
            options.append("6️⃣ ✈️ **查看航空专业知识库**")
        elif question_type == "how_to":
            options.append("6️⃣ 📋 **查看操作指南类文档**")
        elif question_type == "definition":
            options.append("6️⃣ 📚 **查看术语定义类文档**")

        options.append("0️⃣ ❓ **获取系统帮助**（查看使用指南）")

        response += "\n".join(options)
        response += "\n\n💡 提示：直接回复数字即可选择相应操作。"

        return response

    def process_user_choice(self, choice: str, question: str) -> str:
        """处理用户选择的选项"""
        choice = choice.strip()

        if choice == "1" or choice == "1️⃣":
            return "💬 请用更具体、更明确的表述重新提问，例如：\n• '直升机机场的设计标准是什么？'\n• '机场跑道长度有哪些要求？'"

        elif choice == "2" or choice == "2️⃣":
            # 降低阈值重新搜索
            relevant_docs = self.knowledge_base.search(question, similarity_threshold=0.1)
            if relevant_docs:
                return self.format_partial_answer(relevant_docs, question)
            else:
                return "⚠️ 即使降低搜索标准，知识库中仍没有找到相关信息。建议启用网络搜索。"

        elif choice == "3" or choice == "3️⃣":
            # 启用网络搜索
            print("🌐 正在搜索网络信息，请稍候...")
            web_result = self.web_search.search_web(question)
            if web_result:
                return f"🌐 **网络搜索结果** (模型: {web_result.get('model', '未知')}):\n\n{web_result['content']}\n\n---\n*来源：硅基流动网络搜索*"
            else:
                return "❌ 网络搜索失败。请检查网络连接或稍后重试。"

        elif choice == "4" or choice == "4️⃣":
            topics = self.get_available_topics()
            return f"📚 **知识库包含以下主题**：\n\n" + "\n".join(
                [f"• {topic}" for topic in topics]) + "\n\n💡 您可以针对这些主题提问。"

        elif choice == "5" or choice == "5️⃣":
            keywords = self.extract_keywords(question)
            return f"🔑 **建议搜索关键词**：\n\n" + "\n".join([f"• `{kw}`" for kw in keywords]) + "\n\n💡 您可以使用这些关键词重新搜索。"

        elif choice == "6" or choice == "6️⃣":
            q_type = self.classify_question(question)
            if q_type == "aviation_specific":
                return "✈️ **航空专业知识库**：\n\n1. 机场设计标准\n2. 直升机运行规范\n3. 飞行安全规则\n4. 空管通信流程\n5. 应急处理程序\n\n请告诉我您想了解的具体内容。"
            elif q_type == "how_to":
                return "📋 **操作指南类文档**：\n\n1. 机场建设步骤\n2. 直升机起降操作\n3. 安全检查流程\n4. 设备维护方法\n5. 应急处置程序"
            elif q_type == "definition":
                return "📚 **术语定义类文档**：\n\n1. 航空术语表\n2. 技术参数定义\n3. 法规标准解释\n4. 专业名词释义"
            else:
                return "6️⃣ 选项已选择，请具体说明您需要哪方面的帮助。"

        elif choice == "0" or choice == "0️⃣":
            return self.get_help_info()

        else:
            return "❌ 无效的选择。请回复数字1-6或0选择相应操作。"

    def get_help_info(self) -> str:
        """获取系统帮助信息"""
        doc_count = self.knowledge_base.collection.count()

        help_text = f"""
🤖 **智能问答系统 v1.0 - 帮助指南**

**系统状态**
• 📊 知识库文档数: {doc_count} 个片段
• 🌐 网络搜索: {'✅ 已启用' if hasattr(self, 'web_search') else '❌ 未启用'}
• 💾 对话历史: {len(self.conversation_history) // 2} 轮对话

**主要功能**
1. **本地知识库问答** - 基于您提供的机场/直升机文档
2. **智能网络搜索** - 使用硅基流动API获取最新信息
3. **多级响应系统** - 根据匹配程度提供不同级别的回答
4. **会话记忆** - 保持对话上下文

**使用技巧**
• 提问尽量**具体明确**，避免模糊表述
• 使用**完整的问题句式**，如"什么是直升机机场的设计标准？"
• 对于复杂问题，可以**分步骤提问**
• 善用系统提供的**选项菜单**引导搜索
• 输入"退出"或"quit"结束对话

**支持的问题类型**
• ✈️ 航空专业问题（机场、直升机等）
• 📋 操作指南类问题
• 📚 定义解释类问题
• 🔍 事实查询类问题
• 🔄 比较分析类问题

**常用命令**
• 帮助 - 显示此帮助信息
• 状态 - 显示系统状态
• 主题 - 查看知识库主题
• 历史 - 查看对话历史
• 清除 - 清除当前对话历史

**问题反馈**
如有问题或建议，请记录下您的提问和系统的响应。
        """
        return help_text

    def ask(self, question: str) -> str:
        """
        主问答接口

        Args:
            question: 用户问题

        Returns:
            回答内容
        """
        # 清理输入
        question = question.strip()

        if not question:
            return "请提出您的问题。"

        # 检查特殊命令
        if question.lower() in ['帮助', 'help', '?']:
            return self.get_help_info()
        elif question.lower() in ['状态', 'status']:
            return f"📊 系统状态：知识库片段数={self.knowledge_base.collection.count()}, 对话轮数={len(self.conversation_history) // 2}"
        elif question.lower() in ['主题', 'topics', '目录']:
            topics = self.get_available_topics()
            return f"📚 知识库主题：\n" + "\n".join([f"• {t}" for t in topics])
        elif question.lower() in ['历史', 'history']:
            if len(self.conversation_history) > 0:
                history_text = "📝 对话历史：\n"
                for i, entry in enumerate(self.conversation_history[-10:]):  # 显示最后10条
                    role = "👤" if entry.get('role') == 'user' else "🤖"
                    history_text += f"{i + 1}. {role}: {entry.get('content', '')[:50]}...\n"
                return history_text
            else:
                return "📝 当前没有对话历史。"
        elif question.lower() in ['清除', 'clear', '重置']:
            self.conversation_history = []
            self.fallback_count = 0
            return "✅ 对话历史已清除。"

        # 记录对话历史
        self.conversation_history.append({
            "role": "user",
            "content": question,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })

        # 检查是否为选项选择
        if len(self.conversation_history) >= 2:
            last_response = self.conversation_history[-2].get("content", "")
            if "请回复数字" in last_response or "请回复" in last_response:
                # 用户在选择菜单选项
                return self.process_user_choice(question, "")

        print(f"🔍 正在搜索本地知识库...")

        # 1. 搜索本地知识库
        relevant_docs = self.knowledge_base.search(question)

        # 2. 分析结果
        if relevant_docs:
            # 计算最高相似度
            max_similarity = max([doc.get('similarity', 0) for doc in relevant_docs])

            if max_similarity > 0.7:
                # 高置信度结果
                self.fallback_count = 0
                best_doc = max(relevant_docs, key=lambda x: x.get('similarity', 0))
                response = self.format_answer(best_doc)

            elif max_similarity > 0.4:
                # 中等置信度结果
                self.fallback_count = min(self.fallback_count + 1, self.max_fallback_before_escalation)
                response = self.format_partial_answer(relevant_docs, question)

            else:
                # 低置信度结果
                self.fallback_count += 1
                if self.fallback_count >= self.max_fallback_before_escalation:
                    response = f"⚠️ 多次搜索未找到满意答案。\n\n" + self.generate_smart_options(question)
                else:
                    response = self.generate_smart_options(question)

        else:
            # 没有找到结果
            self.fallback_count += 1

            if self.fallback_count >= self.max_fallback_before_escalation:
                response = f"❌ 连续{self.fallback_count}次未找到答案。\n\n" + self.generate_smart_options(question)
            else:
                response = self.generate_smart_options(question)

        # 记录系统响应
        self.conversation_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })

        return response


# ==================== 主程序 ====================

def main():
    """主程序入口"""
    print("\n" + "=" * 60)
    print("🚀 机场-直升机智能问答系统 v1.0")
    print("=" * 60)

    # 配置信息
    KNOWLEDGE_BASE_PATH = r"D:\AlgorithmClub\Damoxingyuanli\homework\datas\附件14 机场 — 直升机场 _Volume II\index.md"
    SILICON_FLOW_API_KEY = "sk-bdgrimfksplnwstzulxfsrdijhjqribunforxvknatzpjlui"

    # 验证路径
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        print(f"❌ 错误：文件路径不存在: {KNOWLEDGE_BASE_PATH}")
        print("请检查文件路径是否正确")
        return

    # 创建问答系统实例
    try:
        qa_system = SmartQASystem(KNOWLEDGE_BASE_PATH, SILICON_FLOW_API_KEY)
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        return

    # 交互循环
    conversation_count = 0

    while True:
        try:
            user_input = input(f"\n👤 第{conversation_count + 1}轮提问: ").strip()
            conversation_count += 1

            if user_input.lower() in ['退出', 'quit', 'exit', 'bye', '再见']:
                print("\n🤖 谢谢使用！再见！")
                break

            if not user_input:
                continue

            # 获取回答
            response = qa_system.ask(user_input)
            print(f"\n🤖 回答: {response}")

        except KeyboardInterrupt:
            print("\n\n👋 对话已中断。")
            break
        except Exception as e:
            print(f"\n❌ 系统错误: {str(e)}")
            print("请重新提问或检查系统配置。")


if __name__ == "__main__":
    main()