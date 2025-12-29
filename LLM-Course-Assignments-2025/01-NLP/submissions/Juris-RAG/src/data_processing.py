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
        from langchain.schema import Document
    except ImportError:
        from langchain_classic.schema import Document
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
    CAIL_CASE_LIMIT = 20000
    STATUTE_SEPARATORS = ["\n第", "\n\n", "\n", "。", "；"]
    EMBED_RPM_LIMIT = 2000
    EMBED_TPM_LIMIT = 500000
    EMBED_BATCH_SIZE = 20
    EMBED_SLEEP_SECONDS = 0.1
    EMBED_MAX_RETRIES = 5
    EMBED_BACKOFF_SECONDS = 10
    EMBED_BACKOFF_MAX_SECONDS = 120
    
    def get_cail_file_path():
        from pathlib import Path
        data_dir = Path(DATA_PATH)
        trimmed = data_dir / "cail_cases_20k.json"
        original = data_dir / "cail_cases.json"
        return str(trimmed if trimmed.exists() else original)


class LegalDataProcessor:
    """法律数据处理器"""
    
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
    
    def load_statutes(self, file_path: str) -> List[Document]:
        """
        加载法条数据
        支持按条款进行智能分割
        """
        print(f"📄 正在加载法条: {file_path}")
        docs = []
        
        if not os.path.exists(file_path):
            print(f"⚠️ 文件 {file_path} 不存在，跳过。")
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 清洗文本
        text = self.clean_text(text)
        
        # 尝试按法条编号分割
        # 匹配 "第X条" 或 "第XX条" 格式
        article_pattern = r'(第[一二三四五六七八九十百千零\d]+条[之的]?[一二三四五六七八九十]*)'
        
        # 使用分割器分割
        chunks = self.text_splitter.split_text(text)
        
        for i, chunk in enumerate(chunks):
            # 提取法条编号（如果有）
            article_match = re.search(article_pattern, chunk)
            article_num = article_match.group(1) if article_match else f"段落{i+1}"
            
            doc_id = self.generate_doc_id(chunk, "刑法")
            
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "source": "中华人民共和国刑法",
                    "type": "statute",
                    "article": article_num,
                    "doc_id": doc_id,
                    "chunk_index": i
                }
            ))
        
        print(f"✅ 加载法条完成，共 {len(docs)} 个文档块")
        return docs
    
    def load_cail_cases(self, file_path: str, limit: int = None) -> List[Document]:
        """
        加载CAIL案例数据
        提取案情事实、罪名、相关法条等信息
        """
        limit = limit or CAIL_CASE_LIMIT
        print(f"⚖️ 正在加载 CAIL 案例: {file_path} (限制 {limit} 条)")
        docs = []
        
        if not os.path.exists(file_path):
            print(f"⚠️ 文件 {file_path} 不存在，跳过。")
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, desc="加载案例", total=limit)):
                if line_num >= limit:
                    break
                
                try:
                    data = json.loads(line)
                    
                    # 提取案情事实
                    fact = data.get('fact', '')
                    if not fact or len(fact) < 50:  # 过滤过短的案情
                        continue
                    
                    fact = self.clean_text(fact)
                    
                    # 提取元数据
                    meta = data.get('meta', {})
                    accusation = meta.get('accusation', [])
                    relevant_articles = meta.get('relevant_articles', [])
                    term_of_imprisonment = meta.get('term_of_imprisonment', {})
                    
                    # 构造结构化内容
                    case_content = f"【案情事实】\n{fact}"
                    
                    # 如果有判决结果，也加入
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
                    print(f"⚠️ 处理第 {line_num} 行时出错: {e}")
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
        构建向量数据库
        使用批量处理避免API超时
        """
        if not docs:
            raise ValueError("❌ 没有文档可供向量化！")
        
        print(f"📦 准备向量化 {len(docs)} 条文档...")
        
        # 确保目录存在
        os.makedirs(DB_PATH, exist_ok=True)
        
        # 删除旧数据库（如果存在）
        if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
            import shutil
            print("🗑️ 清理旧的向量数据库...")
            shutil.rmtree(DB_PATH)
            os.makedirs(DB_PATH)
        
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
                            persist_directory=DB_PATH
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


def build_vector_db():
    """主函数：构建向量数据库"""
    processor = LegalDataProcessor()
    
    # 1. 加载各类数据
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
        build_vector_db()
