import fitz  # 这是 PyMuPDF 的库名
import os


def extract_text_from_pdf(pdf_path, output_txt_path):
    """
    功能：把 PDF 书变成 txt 纯文本
    """
    print(f"正在读取: {pdf_path} ...")

    # 打开 PDF
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"❌ 无法打开文件: {e}")
        return

    # 创建一个 txt 文件准备写入
    with open(output_txt_path, "w", encoding="utf-8") as f:
        # 遍历每一页
        for page_num, page in enumerate(doc):
            # 获取页面文本
            text = page.get_text()
            # 写入 txt，每页之间加个分隔符，方便以后查页码
            f.write(f"\n--- 第 {page_num + 1} 页 ---\n")
            f.write(text)

    print(f"✅ 转换完成！文本已保存到: {output_txt_path}")


if __name__ == "__main__":
    # 定义文件路径
    # 1. 教材
    pdf_file = "data/workbook.pdf"
    txt_file = "data/workbook.txt"

    # 检查文件是否存在
    if os.path.exists(pdf_file):
        extract_text_from_pdf(pdf_file, txt_file)
    else:
        print(f"❌ 找不到文件: {pdf_file}，请检查 data 文件夹！")