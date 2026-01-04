import os
import fitz  # PyMuPDF
from rapidocr_onnxruntime import RapidOCR
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ================= 配置区域 =================
PDF_PATH = "data/textbook.pdf"
DB_PATH = "vector_db"


PAGE_OFFSET = 10


# ===========================================

def load_pdf_with_offset(file_path):
    print(f"🚀 [1/3] 正在加载: {file_path}")
    print(f"    ℹ️ 已启用页码修正: PDF页码 - {PAGE_OFFSET} = 书本页码")

    docs = []
    ocr = RapidOCR()

    with fitz.open(file_path) as pdf:
        total = len(pdf)
        print(f"    - 检测到 PDF 共 {total} 页")

        for i, page in enumerate(pdf):
            # ------------------------------------------------
            # 核心修正逻辑
            # ------------------------------------------------
            physical_page = i + 1  # PDF文件的第几张纸
            logical_page = physical_page - PAGE_OFFSET  # 修正后的书本页码

            # 如果是前 10 页（目录、前言等），显示为 "前言-xx"
            if logical_page <= 0:
                page_label = f"前言/目录"
            else:
                page_label = f"{logical_page}"
            # ------------------------------------------------

            # 1. 尝试直接提取文字
            text = page.get_text()

            # 2. OCR 补救（防止扫描版读不出字）
            if len(text.strip()) < 5:
                try:
                    pix = page.get_pixmap()
                    img_data = pix.tobytes("png")
                    result, _ = ocr(img_data)
                    if result:
                        text = "\n".join([line[1] for line in result])
                except:
                    pass

            # 3. 存入 Document
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source": os.path.basename(file_path),
                        # 这里存入修正后的页码
                        "page": page_label
                    }
                ))

                # 打印日志让我们安心
                if physical_page == 11:
                    print(f"      > ✅ 验证点：PDF第11页 已标记为 -> 第 {page_label} 页")
                elif physical_page % 50 == 0:
                    print(f"      > 处理中：PDF第{physical_page}页 -> 第 {page_label} 页")

    return docs


def main():
    # 1. 加载
    docs = load_pdf_with_offset(PDF_PATH)
    print(f"✅ 提取完成，共 {len(docs)} 页有效内容。")

    # 2. 切分
    print("✂️ [2/3] 正在切分文本...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    # 3. 存入
    print("💾 [3/3] 正在重建向量数据库...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    db = FAISS.from_documents(splits, embeddings)
    db.save_local(DB_PATH)
    print("🎉 数据库重建完毕！现在页码应该完全对上了。")


if __name__ == "__main__":
    main()