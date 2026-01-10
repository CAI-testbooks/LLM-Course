#!/usr/bin/env python3
"""
处理纯文本版Python官方文档 (python-3.14-docs-text)
将文档分块并保存为JSON格式，供RAG系统使用
"""
import os
import json
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_text_files(doc_path):
    """
    加载纯文本文件
    Args:
        doc_path: 文本文档目录路径，例如 "./data/python-3.14-docs-text"
    Returns:
        包含所有文档内容的列表
    """
    raw_texts = []
    file_count = 0

    print(f"开始从目录加载文档: {doc_path}")

    # 遍历目录下的所有.txt文件
    for root, dirs, files in os.walk(doc_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read().strip()

                        # 过滤掉空文件或极短的文件
                        if len(content) > 100:
                            # 添加文件来源信息
                            rel_path = os.path.relpath(file_path, doc_path)
                            formatted_content = f"【来源: {rel_path}】\n{content}"
                            raw_texts.append(formatted_content)
                            file_count += 1

                            # 显示进度
                            if file_count % 100 == 0:
                                print(f"  已加载 {file_count} 个文件...")
                except Exception as e:
                    print(f"  警告: 无法读取文件 {file_path}: {e}")
                    continue

    print(f" 成功从 {file_count} 个文本文件中加载了内容")
    return raw_texts


def clean_text_content(texts):
    """
    清理文本内容
    Args:
        texts: 原始文本列表
    Returns:
        清理后的文本列表
    """
    cleaned_texts = []


    for i, text in enumerate(texts):
        # 1. 移除过长的空白行（保留正常的段落分隔）
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

        # 2. 移除ASCII艺术或装饰线
        text = re.sub(r'^[-=*_]{10,}$', '', text, flags=re.MULTILINE)

        # 3. 移除过短的段落（可能是目录项或页眉页脚）
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            # 保留有意义的长行或来源标记
            if len(line) > 30 or line.startswith('【来源:') or 'Copyright' in line:
                cleaned_lines.append(line)

        cleaned_text = '\n'.join(cleaned_lines)

        # 4. 只保留足够长的文档
        if len(cleaned_text) > 200:
            cleaned_texts.append(cleaned_text)

        # 显示进度
        if (i + 1) % 200 == 0:
            print(f"  已清理 {i + 1}/{len(texts)} 个文档...")

    print(f"清理完成，保留 {len(cleaned_texts)} 个有效文档")
    return cleaned_texts


def split_documents_optimized(texts):
    """
    为Python文档优化的分块策略
    Args:
        texts: 清理后的文本列表
    Returns:
        分块后的文档列表
    """

    # 针对Python文档优化的分块器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # 稍大的块，因为Python文档通常结构清晰
        chunk_overlap=100,  # 适当的重叠保持上下文
        separators=[  # Python文档特定的分隔符
            "\n\n\n",  # 主要章节分隔
            "\n\n",  # 段落分隔
            "\n• ",  # 列表项
            "\n",  # 换行符
            " ",  # 空格
            ""  # 最后的手段
        ],
        length_function=len,
        keep_separator=True  # 保留分隔符有助于理解结构
    )

    all_chunks = []

    for i, text in enumerate(texts):
        try:
            chunks = text_splitter.split_text(text)
            all_chunks.extend(chunks)

            # 显示进度
            if (i + 1) % 100 == 0:
                print(f"  已处理 {i + 1}/{len(texts)} 个文档，生成 {len(all_chunks)} 个块...")
        except Exception as e:
            print(f"  警告: 处理文档 {i} 时分块失败: {e}")
            # 如果分块失败，尝试简单分割
            simple_chunks = [text[j:j + 500] for j in range(0, len(text), 500)]
            all_chunks.extend(simple_chunks)

    print(f" 分块完成！共生成 {len(all_chunks)} 个文本块")

    # 显示一些样本
    print("\n📋 文本块样本预览:")
    for i in range(min(3, len(all_chunks))):
        print(f"\n--- 块 {i + 1} (前200字符) ---")
        print(all_chunks[i][:200] + "...")

    return all_chunks


def save_chunks_to_json(chunks, output_path):
    """
    保存分块结果到JSON文件
    Args:
        chunks: 文本块列表
        output_path: 输出文件路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 保存为JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f" 文本块已保存至: {output_path}")

    # 计算统计信息
    total_chars = sum(len(chunk) for chunk in chunks)
    avg_chunk_size = total_chars / len(chunks) if chunks else 0

    print(f" 统计信息:")
    print(f"  - 文本块数量: {len(chunks)}")
    print(f"  - 总字符数: {total_chars:,}")
    print(f"  - 平均块大小: {avg_chunk_size:.1f} 字符")
    print(f"  - 最小块: {min(len(c) for c in chunks) if chunks else 0} 字符")
    print(f"  - 最大块: {max(len(c) for c in chunks) if chunks else 0} 字符")


def main():
    """主处理函数"""
    print("=" * 60)
    print("Python 3.14 文档处理工具")
    print("=" * 60)

    # 1. 配置路径
    # 注意：这里假设你的文件夹名为 python-3.14-docs-text
    # 如果文件夹名不同，请修改这里
    doc_path = "./data/python-3.14-docs-text"
    output_path = "./data/document_chunks.json"

    # 检查目录是否存在
    if not os.path.exists(doc_path):
        print(f" 错误: 文档目录不存在: {doc_path}")
        return

    print(f"文档目录: {doc_path}")
    print(f"输出文件: {output_path}")
    print("-" * 60)

    # 2. 加载文档
    raw_texts = load_text_files(doc_path)

    if not raw_texts:
        print(" 错误: 未找到任何文本文件")
        return

    # 3. 清理文档
    cleaned_texts = clean_text_content(raw_texts)

    # 4. 分块处理
    chunks = split_documents_optimized(cleaned_texts)

    if not chunks:
        print(" 错误: 分块后没有生成任何文本块")
        return

    # 5. 保存结果
    save_chunks_to_json(chunks, output_path)


if __name__ == "__main__":
    main()