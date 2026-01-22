from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import re


# 从每条记录中提取元数据 标题、发表日期
def metadata_func(record: dict, metadata: dict) -> dict:
    pub_date = record.get("pub_date", {})
    metadata["title"] = record.get("article_title", "")
    metadata["year"] = pub_date.get("year", "")
    metadata["month"] = pub_date.get("month", "")
    metadata["day"] = pub_date.get("day", "")
    return metadata


# 清洗文本中的特殊字符和冗余信息
def clean_text(text: str) -> str:
    if not text:
        return ""
    # 去除HTML标签和特殊符号
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\w\s.,!?;:-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# 加载并处理PubMed数据
def load_and_process_pubmed(json_path: str) -> list[Document]:
    # 初始化JSONLoader
    loader = JSONLoader(
        file_path=json_path,
        jq_schema=".[]",
        content_key="article_abstract",
        metadata_func=metadata_func
    )

    # 加载原始数据
    raw_documents = loader.load()
    print(f"原始数据加载完成，共 {len(raw_documents)} 篇文献")

    # 清洗文本内容
    cleaned_documents = []
    for doc in raw_documents:
        cleaned_content = clean_text(doc.page_content)
        cleaned_doc = Document(
            page_content=cleaned_content,
            metadata=doc.metadata
        )
        cleaned_documents.append(cleaned_doc)

    print(f"文本清洗完成，有效文献数量：{len(cleaned_documents)}")
    return cleaned_documents


# 使用TokenTextSplitter对文档进行分块
def split_documents(documents: list[Document], chunk_size: int = 128, chunk_overlap: int = 64) -> list[Document]:
    text_splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap  # 块之间的重叠token数量
    )
    chunks = text_splitter.split_documents(documents)
    print(f"文本分块完成：{len(documents)} 篇文献 → {len(chunks)} 个片段")
    return chunks


# 初始化词嵌入
def init_embeddings(model_name: str = "all-mpnet-base-v2", device: str = "cuda") -> HuggingFaceEmbeddings:
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": False}
    )
    print(f"嵌入模型初始化完成：{model_name}（运行设备：{device}）")
    return embeddings


# 构建FAISS向量数据库
def build_vector_db(chunks: list[Document], embeddings: HuggingFaceEmbeddings, save_path: str = "pubmed_faiss_db") -> FAISS:
    db = FAISS.from_documents(chunks, embeddings)
    # 保存数据库到本地（方便后续复用）
    db.save_local(save_path)
    print(f"向量数据库构建完成，已保存至 {save_path}")
    return db
