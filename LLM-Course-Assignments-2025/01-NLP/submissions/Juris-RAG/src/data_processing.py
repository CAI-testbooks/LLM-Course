"""
Juris-RAG 数据处理模块
负责法律文本的加载、清洗、分块和向量化
"""
import os
import re
import json
import time
import hashlib
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

# LangChain 组件
try:
    from langchain_core.documents import Document
except ImportError:  # fallback for older langchain versions
    try:
        from langchain_core.documents import Document
    except ImportError:
        from langchain_classic.schema import Document

# 优先使用新版 langchain-chroma
try:
    from langchain_chroma import Chroma  # type: ignore
except ImportError:  # pragma: no cover - 兼容旧环境
    from langchain_community.vectorstores import Chroma

try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:  # fallback for older installs
    from langchain_community.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 导入配置
try:
    from src.config import (
        DATA_PATH, DB_PATH, EMBEDDING_MODEL, SILICONFLOW_API_KEY,
        SILICONFLOW_BASE_URL, CHUNK_SIZE, CHUNK_OVERLAP, 
        CAIL_CASE_LIMIT, STATUTE_SEPARATORS,
        EMBED_RPM_LIMIT, EMBED_TPM_LIMIT,
        EMBED_BATCH_SIZE, EMBED_SLEEP_SECONDS, EMBED_MAX_RETRIES,
        EMBED_BACKOFF_SECONDS, EMBED_BACKOFF_MAX_SECONDS
    )
    from src.cail_adapter import get_cail_file_path
except ImportError:
    # 默认配置
    DATA_PATH = "./data/raw"
    DB_PATH = "./data/vector_db"
    EMBEDDING_MODEL = "BAAI/bge-m3"
    SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
    SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    CAIL_CASE_LIMIT = int(os.getenv("CAIL_CASE_LIMIT", "100000"))
    STATUTE_SEPARATORS = ["\n第", "\n\n", "\n", "。", "；"]
    EMBED_RPM_LIMIT = int(os.getenv("EMBED_RPM_LIMIT", "2000"))
    EMBED_TPM_LIMIT = int(os.getenv("EMBED_TPM_LIMIT", "500000"))
    EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "20"))
    EMBED_SLEEP_SECONDS = float(os.getenv("EMBED_SLEEP_SECONDS", "0.1"))
    EMBED_MAX_RETRIES = 5
    EMBED_BACKOFF_SECONDS = 10
    EMBED_BACKOFF_MAX_SECONDS = 120
    
    def get_cail_file_path():
        from pathlib import Path
        data_dir = Path(DATA_PATH)
        cail_file = data_dir / "cail_cases.json"
        return str(cail_file)


class LegalDataProcessor:
    """法律数据处理器 - 支持多领域法律数据"""
    
    # 定义支持的法律领域
    LEGAL_DOMAINS = {
        'criminal': {
            'name': '刑法',
            'file': 'criminal_code.txt',
            'priority': 1,  # 最高优先级
            'description': '中华人民共和国刑法 - 关于犯罪和刑罚的规定'
        },
        'civil': {
            'name': '民法',
            'file': 'civil_code.txt',
            'priority': 2,
            'description': '中华人民共和国民法典 - 关于民事权利和义务的规定'
        },
        'commercial': {
            'name': '商法',
            'file': 'commercial_law.txt',
            'priority': 3,
            'description': '商法及相关法律 - 关于商业、公司、证券等的规定'
        },
        'administrative': {
            'name': '行政法',
            'file': 'administrative_law.txt',
            'priority': 4,
            'description': '行政法及相关法律 - 关于行政行为和行政管理的规定'
        },
        'labor': {
            'name': '劳动法',
            'file': 'labor_law.txt',
            'priority': 5,
            'description': '劳动法及相关法律 - 关于劳动权利和义务的规定'
        }
    }
    
    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key or SILICONFLOW_API_KEY
        self.base_url = base_url or SILICONFLOW_BASE_URL
        
        if not self.api_key:
            raise ValueError("❌ 未找到 SILICONFLOW_API_KEY，请检查环境变量！")
        
        # 初始化Embedding模型
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_base=self.base_url,
            openai_api_key=self.api_key
        )
        
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=STATUTE_SEPARATORS,
            length_function=len
        )
        
        # 存储不同领域的向量库
        self.vectorstores = {}
    
    def clean_text(self, text: str) -> str:
        """清洗文本：去除多余空白、特殊字符等"""
        if not text:
            return ""
        
        # 替换多个空格/换行为单个
        text = re.sub(r'\s+', ' ', text)
        # 去除特殊控制字符
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        # 规范化标点
        text = text.replace('．', '.').replace('，', '，').replace('。', '。')
        
        return text.strip()
    
    def generate_doc_id(self, content: str, source: str) -> str:
        """生成文档唯一ID"""
        hash_input = f"{source}:{content[:100]}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def load_statutes(self, file_path: str, domain_key: str = 'criminal') -> List[Document]:
        """
        加载法条数据 - 按单个条款进行细粒度分块
        每个条款单独作为一个文档，确保检索精度
        
        Args:
            file_path: 法律文本文件路径
            domain_key: 法律领域标识符（如'criminal', 'civil'等）
        """
        domain_info = self.LEGAL_DOMAINS.get(domain_key, {})
        domain_name = domain_info.get('name', domain_key)
        
        print(f"📄 正在加载 {domain_name} 法条: {file_path}")
        docs = []
        
        if not os.path.exists(file_path):
            print(f"⚠️ 文件 {file_path} 不存在，跳过。")
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 预处理：规范化空白字符
        text = re.sub(r'　', ' ', text)  # 全角空格转半角
        text = re.sub(r'\t', ' ', text)
        
        # 按条款分割 - 使用正则匹配 "第X条" 开头的段落
        # 匹配格式：第X条 或 第X条之X
        article_pattern = r'(第[一二三四五六七八九十百千零\d]+条(?:之[一二三四五六七八九十]*)?)\s*'
        
        # 分割文本，保留分隔符
        parts = re.split(f'({article_pattern})', text)
        
        current_article = None
        current_content = []
        chapter_info = ""  # 当前章节信息
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # 检测章节标题（如 "第四章 侵犯公民人身权利、民主权利罪"）
            chapter_match = re.match(r'(第[一二三四五六七八九十]+(?:编|章|节)[^\n]*)', part)
            if chapter_match:
                chapter_info = chapter_match.group(1)
                continue
            
            # 检测条款号
            article_match = re.match(article_pattern, part)
            if article_match:
                # 保存之前的条款
                if current_article and current_content:
                    content_text = ' '.join(current_content)
                    content_text = self.clean_text(content_text)
                    
                    if len(content_text) > 20:  # 过滤过短的内容
                        doc_id = self.generate_doc_id(content_text, domain_name)
                        
                        # 在内容前添加条款号和章节信息，增强检索效果
                        full_content = f"{current_article} {content_text}"
                        if chapter_info:
                            full_content = f"【{chapter_info}】{full_content}"
                        
                        docs.append(Document(
                            page_content=full_content,
                            metadata={
                                "source": domain_info.get('description', "中华人民共和国刑法"),
                                "domain": domain_key,
                                "domain_name": domain_name,
                                "type": "statute",
                                "article": current_article,
                                "chapter": chapter_info,
                                "doc_id": doc_id,
                                "chunk_index": len(docs)
                            }
                        ))
                
                # 开始新条款
                current_article = article_match.group(1)
                current_content = []
            else:
                # 非条款号的内容，添加到当前条款
                if current_article:
                    current_content.append(part)
                elif not any(kw in part for kw in ['目录', '目　　录', '第一编', '第二编']):
                    # 前言部分（修订历史等），也可以保留
                    if len(part) > 100:
                        doc_id = self.generate_doc_id(part[:100], f"{domain_name}前言")
                        docs.append(Document(
                            page_content=self.clean_text(part),
                            metadata={
                                "source": domain_info.get('description', domain_name),
                                "domain": domain_key,
                                "domain_name": domain_name,
                                "type": "statute",
                                "article": "前言",
                                "chapter": "",
                                "doc_id": doc_id,
                                "chunk_index": len(docs)
                            }
                        ))
        
        # 保存最后一个条款
        if current_article and current_content:
            content_text = ' '.join(current_content)
            content_text = self.clean_text(content_text)
            
            if len(content_text) > 20:
                doc_id = self.generate_doc_id(content_text, "刑法")
                full_content = f"{current_article} {content_text}"
                if chapter_info:
                    full_content = f"【{chapter_info}】{full_content}"
                
                docs.append(Document(
                    page_content=full_content,
                    metadata={
                        "source": domain_info.get('description', domain_name),
                        "domain": domain_key,
                        "domain_name": domain_name,
                        "type": "statute",
                        "article": current_article,
                        "chapter": chapter_info,
                        "doc_id": doc_id,
                        "chunk_index": len(docs)
                    }
                ))
        
        # 对于过长的条款，进行二次分割
        final_docs = []
        max_length = 800  # 单个chunk最大长度
        
        for doc in docs:
            if len(doc.page_content) > max_length:
                # 按句子分割长条款
                sentences = re.split(r'([。；])', doc.page_content)
                
                current_chunk = ""
                chunk_index = 0
                
                for i in range(0, len(sentences), 2):
                    sentence = sentences[i]
                    punct = sentences[i+1] if i+1 < len(sentences) else ""
                    
                    if len(current_chunk) + len(sentence) + len(punct) > max_length:
                        if current_chunk:
                            new_doc = Document(
                                page_content=current_chunk.strip(),
                                metadata={
                                    **doc.metadata,
                                    "article": f"{doc.metadata['article']}(第{chunk_index+1}部分)",
                                    "doc_id": f"{doc.metadata['doc_id']}_{chunk_index}"
                                }
                            )
                            final_docs.append(new_doc)
                            chunk_index += 1
                        current_chunk = sentence + punct
                    else:
                        current_chunk += sentence + punct
                
                if current_chunk.strip():
                    new_doc = Document(
                        page_content=current_chunk.strip(),
                        metadata={
                            **doc.metadata,
                            "article": f"{doc.metadata['article']}(第{chunk_index+1}部分)" if chunk_index > 0 else doc.metadata['article'],
                            "doc_id": f"{doc.metadata['doc_id']}_{chunk_index}" if chunk_index > 0 else doc.metadata['doc_id']
                        }
                    )
                    final_docs.append(new_doc)
            else:
                final_docs.append(doc)
        
        print(f"✅ 加载法条完成，共 {len(final_docs)} 个文档块")
        
        # 打印一些示例
        sample_articles = ["第二百三十二条", "第二百六十四条", "第二十条", "第六十七条"]
        print("📋 关键法条示例:")
        for doc in final_docs[:]:
            if doc.metadata.get("article") in sample_articles:
                print(f"   - {doc.metadata['article']}: {doc.page_content[:60]}...")
        
        return final_docs
    
    # ==================== 方案E: 罪名关键词索引 ====================
    def _extract_crime_keywords(self, content: str) -> List[str]:
        """
        从法条内容中提取罪名关键词
        用于增强检索效果
        """
        crime_patterns = {
            # 侵犯公民人身权利罪
            "故意杀人": ["故意杀人", "杀害", "杀人"],
            "故意伤害": ["故意伤害", "伤害", "轻伤", "重伤"],
            "强奸": ["强奸", "奸淫", "妇女"],
            "绑架": ["绑架", "人质", "勒索"],
            "拐卖": ["拐卖", "收买", "妇女儿童"],
            "非法拘禁": ["非法拘禁", "剥夺人身自由"],
            
            # 侵犯财产罪
            "盗窃": ["盗窃", "偷盗", "窃取"],
            "抢劫": ["抢劫", "暴力劫取", "持械抢劫"],
            "诈骗": ["诈骗", "骗取", "虚构事实"],
            "抢夺": ["抢夺", "公然夺取"],
            "敲诈勒索": ["敲诈勒索", "威胁", "要挟财物"],
            "侵占": ["侵占", "代为保管", "拒不退还"],
            
            # 危害公共安全罪
            "交通肇事": ["交通肇事", "交通事故", "逃逸"],
            "危险驾驶": ["危险驾驶", "醉酒驾驶", "追逐竞驶"],
            "放火": ["放火", "纵火", "危害公共安全"],
            "爆炸": ["爆炸", "炸弹", "危害公共安全"],
            
            # 妨害社会管理秩序罪
            "聚众斗殴": ["聚众斗殴", "斗殴", "首要分子"],
            "寻衅滋事": ["寻衅滋事", "随意殴打", "强拿硬要"],
            "赌博": ["赌博", "开设赌场"],
            "伪证": ["伪证", "虚假证明", "证人"],
            "包庇": ["包庇", "窝藏", "隐瞒"],
            
            # 贪污贿赂罪
            "贪污": ["贪污", "侵吞", "国家工作人员"],
            "受贿": ["受贿", "收受财物", "谋取利益"],
            "行贿": ["行贿", "给予财物"],
            
            # 走私贩毒罪
            "走私": ["走私", "武器弹药", "核材料"],
            "贩毒": ["贩毒", "毒品", "走私贩卖运输"],
            
            # 刑罚制度
            "正当防卫": ["正当防卫", "防卫过当", "不法侵害"],
            "紧急避险": ["紧急避险", "避免危险"],
            "自首": ["自首", "自动投案", "如实供述"],
            "立功": ["立功", "重大立功", "检举揭发"],
            "累犯": ["累犯", "刑罚执行完毕", "五年以内"],
            "缓刑": ["缓刑", "宣告缓刑", "考验期"],
            "减刑": ["减刑", "悔改表现"],
            "假释": ["假释", "服刑期间"],
        }
        
        found_keywords = []
        content_lower = content.lower()
        
        for crime, patterns in crime_patterns.items():
            for pattern in patterns:
                if pattern in content_lower:
                    found_keywords.append(crime)
                    break
        
        return list(set(found_keywords))
    
    def load_cail_cases(self, file_path: str, limit: int = None) -> List[Document]:
        """
        加载CAIL案例数据
        支持两种格式：
        1. JSON数组格式: [{...}, {...}]
        2. JSONL格式: 每行一条JSON
        """
        limit = limit or CAIL_CASE_LIMIT
        print(f"⚖️ 正在加载 CAIL 案例: {file_path} (限制 {limit} 条)")
        docs = []
        
        if not os.path.exists(file_path):
            print(f"⚠️ 文件 {file_path} 不存在，跳过。")
            return []
        
        # 先尝试作为JSON数组格式读取
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if content.startswith('['):
                # JSON数组格式
                print("   [JSON数组格式]")
                cases = json.loads(content)
                
                if isinstance(cases, list):
                    total_cases = len(cases)
                    print(f"   检测到 {total_cases} 条案例")
                    
                    for case_idx, data in enumerate(tqdm(cases[:limit], desc="加载案例", total=min(len(cases), limit))):
                        try:
                            # 处理两种可能的数据结构
                            if isinstance(data, dict):
                                # 结构1: {"fact": "...", "meta": {...}}
                                fact = data.get('fact', '')
                                meta = data.get('meta', {})
                                
                                # 结构2: {"fact": "...", "accusation": [...], "law_articles": [...]}
                                if 'accusation' in data and not meta:
                                    meta = {
                                        'accusation': data.get('accusation', []),
                                        'relevant_articles': data.get('law_articles', []),
                                        'term_of_imprisonment': {
                                            'imprisonment': data.get('prison_term', 0),
                                            'death_penalty': False,
                                            'life_imprisonment': False
                                        },
                                        'punish_of_money': data.get('fine', 0)
                                    }
                            else:
                                continue
                            
                            # 提取案情事实
                            if not fact or len(fact) < 50:
                                continue
                            
                            fact = self.clean_text(fact)
                            
                            # 提取元数据
                            accusation = meta.get('accusation', [])
                            relevant_articles = meta.get('relevant_articles', [])
                            term_of_imprisonment = meta.get('term_of_imprisonment', {})
                            
                            # 构造结构化内容
                            case_content = f"【案情事实】\n{fact}"
                            
                            # 添加罪名信息
                            if accusation:
                                acc_str = "、".join(accusation) if isinstance(accusation, list) else str(accusation)
                                case_content += f"\n【指控罪名】{acc_str}"
                            
                            # 添加法条信息
                            if relevant_articles:
                                articles_str = "、".join(str(a) for a in relevant_articles) if isinstance(relevant_articles, list) else str(relevant_articles)
                                case_content += f"\n【相关法条】第{articles_str}条"
                            
                            # 添加判决结果
                            if term_of_imprisonment:
                                death = term_of_imprisonment.get('death_penalty', False)
                                life = term_of_imprisonment.get('life_imprisonment', False)
                                imprisonment = term_of_imprisonment.get('imprisonment', 0)
                                
                                if death:
                                    sentence = "死刑"
                                elif life:
                                    sentence = "无期徒刑"
                                elif imprisonment > 0:
                                    sentence = f"有期徒刑{imprisonment}个月"
                                else:
                                    sentence = "其他刑罚"
                                
                                case_content += f"\n【判决结果】{sentence}"
                            
                            doc_id = self.generate_doc_id(fact[:100], "CAIL")
                            
                            docs.append(Document(
                                page_content=case_content,
                                metadata={
                                    "source": "CAIL2018司法案例数据集",
                                    "type": "case",
                                    "accusation": ",".join(accusation) if accusation else "未知",
                                    "articles": ",".join(str(a) for a in relevant_articles) if relevant_articles else "未知",
                                    "doc_id": doc_id,
                                    "case_index": case_idx
                                }
                            ))
                        
                        except Exception as e:
                            continue
                    
                    print(f"✅ 加载案例完成，共 {len(docs)} 个文档")
                    return docs
        
        except (json.JSONDecodeError, ValueError):
            pass  # 不是JSON数组格式，尝试JSONL格式
        
        # 如果不是JSON数组格式，尝试JSONL格式（每行一条JSON）
        print("   [JSONL格式]")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, desc="加载案例", total=limit)):
                if line_num >= limit:
                    break
                
                try:
                    data = json.loads(line)
                    
                    # 提取案情事实
                    fact = data.get('fact', '')
                    if not fact or len(fact) < 50:
                        continue
                    
                    fact = self.clean_text(fact)
                    
                    # 提取元数据
                    meta = data.get('meta', {})
                    accusation = meta.get('accusation', [])
                    relevant_articles = meta.get('relevant_articles', [])
                    term_of_imprisonment = meta.get('term_of_imprisonment', {})
                    
                    # 构造结构化内容
                    case_content = f"【案情事实】\n{fact}"
                    
                    if accusation:
                        case_content += f"\n【指控罪名】{'、'.join(accusation)}"
                    
                    if term_of_imprisonment:
                        death = term_of_imprisonment.get('death_penalty', False)
                        life = term_of_imprisonment.get('life_imprisonment', False)
                        imprisonment = term_of_imprisonment.get('imprisonment', 0)
                        
                        if death:
                            sentence = "死刑"
                        elif life:
                            sentence = "无期徒刑"
                        elif imprisonment > 0:
                            sentence = f"有期徒刑{imprisonment}个月"
                        else:
                            sentence = "其他刑罚"
                        
                        case_content += f"\n【判决结果】{sentence}"
                    
                    doc_id = self.generate_doc_id(fact, "CAIL")
                    
                    docs.append(Document(
                        page_content=case_content,
                        metadata={
                            "source": "CAIL2018司法案例数据集",
                            "type": "case",
                            "accusation": ",".join(accusation) if accusation else "未知",
                            "articles": ",".join(str(a) for a in relevant_articles),
                            "doc_id": doc_id,
                            "case_index": line_num
                        }
                    ))
                
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    continue
        
        print(f"✅ 加载案例完成，共 {len(docs)} 个文档")
        return docs
    
    def load_qa_pairs(self, file_path: str) -> List[Document]:
        """
        加载QA对数据（如果有）
        格式: JSON Lines，每行 {"question": "...", "answer": "..."}
        """
        print(f"❓ 正在加载QA对: {file_path}")
        docs = []
        
        if not os.path.exists(file_path):
            print(f"⚠️ 文件 {file_path} 不存在，跳过。")
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line)
                    question = self.clean_text(data.get('question', ''))
                    answer = self.clean_text(data.get('answer', ''))
                    
                    if question and answer:
                        content = f"【问题】{question}\n【回答】{answer}"
                        doc_id = self.generate_doc_id(content, "QA")
                        
                        docs.append(Document(
                            page_content=content,
                            metadata={
                                "source": "法律QA数据集",
                                "type": "qa",
                                "doc_id": doc_id,
                                "qa_index": line_num
                            }
                        ))
                except:
                    continue
        
        print(f"✅ 加载QA对完成，共 {len(docs)} 个文档")
        return docs
    
    def build_vector_db(self, docs: List[Document], batch_size: int = EMBED_BATCH_SIZE) -> Chroma:
        """
        构建向量数据库（使用默认路径）
        使用批量处理避免API超时
        """
        return self.build_vector_db_with_path(docs, DB_PATH, batch_size)
    
    def build_vector_db_with_path(self, docs: List[Document], db_path: str, batch_size: int = EMBED_BATCH_SIZE) -> Chroma:
        """
        构建向量数据库（指定路径）
        用于支持多领域的独立向量库
        
        Args:
            docs: 待向量化的文档列表
            db_path: 数据库保存路径
            batch_size: 批处理大小
        """
        if not docs:
            raise ValueError("❌ 没有文档可供向量化！")
        
        print(f"📦 准备向量化 {len(docs)} 条文档到 {db_path}...")
        
        # 确保目录存在
        os.makedirs(db_path, exist_ok=True)
        
        # 删除旧数据库（如果存在），避免版本冲突
        if os.path.exists(db_path):
            try:
                import shutil
                print(f"🗑️ 清理旧的向量数据库 {db_path}...")
                shutil.rmtree(db_path)
                time.sleep(0.5)  # 等待文件系统完成删除
            except Exception as e:
                print(f"⚠️ 清理旧数据库失败: {e}")
        
        os.makedirs(db_path, exist_ok=True)
        
        vectorstore = None
        
        # 批量处理
        def is_rate_limit_error(err: Exception) -> bool:
            message = str(err).lower()
            return ("rate limit" in message or "rpm limit" in message or "429" in message or "too many" in message)

        def get_batch_sleep_seconds(batch_docs) -> float:
            rpm_wait = 0.0
            if EMBED_RPM_LIMIT > 0:
                rpm_wait = 60.0 / EMBED_RPM_LIMIT
            tpm_wait = 0.0
            if EMBED_TPM_LIMIT > 0:
                approx_tokens = sum(len(doc.page_content) for doc in batch_docs)
                tpm_wait = (approx_tokens / EMBED_TPM_LIMIT) * 60.0
            return max(EMBED_SLEEP_SECONDS, rpm_wait, tpm_wait)

        for i in tqdm(range(0, len(docs), batch_size), desc="向量化进度"):
            batch = docs[i:i + batch_size]
            retries = 0
            
            while True:
                try:
                    if vectorstore is None:
                        # 第一批：创建新的向量库
                        vectorstore = Chroma.from_documents(
                            documents=batch,
                            embedding=self.embeddings,
                            persist_directory=db_path
                        )
                    else:
                        # 后续批：添加到现有向量库
                        vectorstore.add_documents(batch)
                    
                    # 避免API速率限制
                    sleep_seconds = get_batch_sleep_seconds(batch)
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)
                    break
                    
                except Exception as e:
                    if is_rate_limit_error(e):
                        retries += 1
                        if retries > EMBED_MAX_RETRIES:
                            raise RuntimeError(
                                "触发RPM限制，已达到最大重试次数。"
                                "请完成账号实名认证或增大等待时间后重试。"
                            ) from e
                        backoff = min(EMBED_BACKOFF_SECONDS * (2 ** (retries - 1)), EMBED_BACKOFF_MAX_SECONDS)
                        print(f"⚠️ 批次 {i//batch_size + 1} 触发限速，等待 {backoff:.0f}s 后重试...")
                        time.sleep(backoff)
                        continue
                    
                    print(f"⚠️ 批次 {i//batch_size + 1} 处理失败: {e}")
                    time.sleep(2)  # 出错后多等待
                    break
        
        print(f"✅ 向量数据库构建完成！已保存至 {DB_PATH}")
        return vectorstore
    
    def get_statistics(self, docs: List[Document]) -> Dict:
        """获取数据集统计信息"""
        stats = {
            "total_docs": len(docs),
            "by_type": {},
            "avg_length": 0,
            "total_chars": 0
        }
        
        for doc in docs:
            doc_type = doc.metadata.get("type", "unknown")
            stats["by_type"][doc_type] = stats["by_type"].get(doc_type, 0) + 1
            stats["total_chars"] += len(doc.page_content)
        
        stats["avg_length"] = stats["total_chars"] / len(docs) if docs else 0
        
        return stats


def build_vector_db(multi_domain: bool = True):
    """主函数：构建向量数据库（支持单领域或多领域）
    
    Args:
        multi_domain: 是否使用多领域系统，默认为True
    """
    processor = LegalDataProcessor()
    
    if multi_domain:
        print("\n🌍 多领域模式启动")
        print("="*60)
        
        # 为每个领域建立独立的向量库
        for domain_key, domain_info in processor.LEGAL_DOMAINS.items():
            domain_name = domain_info['name']
            domain_file = domain_info['file']
            
            file_path = os.path.join(DATA_PATH, domain_file)
            
            if not os.path.exists(file_path):
                print(f"⚠️  {domain_name} 文件不存在: {file_path}")
                continue
            
            print(f"\n📚 加载 {domain_name}...")
            
            domain_docs = []
            
            # 加载该领域的法律文本
            domain_docs.extend(processor.load_statutes(file_path, domain_key=domain_key))
            
            # 仅为刑法领域加载案例和QA对
            if domain_key == 'criminal':
                cail_path = get_cail_file_path()
                domain_docs.extend(processor.load_cail_cases(cail_path, limit=CAIL_CASE_LIMIT))
                
                qa_path = os.path.join(DATA_PATH, "legal_qa.json")
                if os.path.exists(qa_path):
                    domain_docs.extend(processor.load_qa_pairs(qa_path))
            
            if domain_docs:
                stats = processor.get_statistics(domain_docs)
                print(f"  ✅ {domain_name}: {stats['total_docs']} 文档, {stats['total_chars']:,} 字符")
                
                # 为该领域创建独立的向量库
                domain_db_path = os.path.join(DB_PATH, domain_key)
                processor.build_vector_db_with_path(domain_docs, domain_db_path)
                processor.vectorstores[domain_key] = domain_db_path
            else:
                print(f"  ⚠️  {domain_name}: 无数据")
        
        print("\n✅ 多领域向量数据库构建完成！")
        print(f"   已构建的领域: {list(processor.vectorstores.keys())}")
        
    else:
        # 单领域模式（保留向后兼容性）
        print("\n🔍 单领域模式启动 - 仅加载刑法")
        print("="*60)
        
        all_docs = []
        
        # 加载法条
        statute_path = os.path.join(DATA_PATH, "criminal_code.txt")
        statute_docs = processor.load_statutes(statute_path)
        all_docs.extend(statute_docs)
        
        # 加载CAIL案例
        cail_path = get_cail_file_path()
        case_docs = processor.load_cail_cases(cail_path, limit=CAIL_CASE_LIMIT)
        all_docs.extend(case_docs)
        
        # 加载QA对（如果存在）
        qa_path = os.path.join(DATA_PATH, "legal_qa.json")
        if os.path.exists(qa_path):
            qa_docs = processor.load_qa_pairs(qa_path)
            all_docs.extend(qa_docs)
        
        if not all_docs:
            print("❌ 没有加载到任何数据，请检查 data/raw 目录。")
            return
        
        # 2. 打印统计信息
        stats = processor.get_statistics(all_docs)
        print("\n📊 数据集统计:")
        print(f"   总文档数: {stats['total_docs']}")
        print(f"   按类型分布: {stats['by_type']}")
        print(f"   平均长度: {stats['avg_length']:.1f} 字符")
        print(f"   总字符数: {stats['total_chars']:,}")
        
        # 3. 构建向量数据库
        processor.build_vector_db(all_docs)


if __name__ == "__main__":
    if not SILICONFLOW_API_KEY:
        print("❌ 请先设置 SILICONFLOW_API_KEY 环境变量！")
        print("   Windows: set SILICONFLOW_API_KEY=your_key")
        print("   Linux/Mac: export SILICONFLOW_API_KEY=your_key")
    else:
        build_vector_db(multi_domain=True)  # 启用多领域模式
