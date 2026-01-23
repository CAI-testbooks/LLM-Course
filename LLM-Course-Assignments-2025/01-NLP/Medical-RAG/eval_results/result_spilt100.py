import json
from typing import List, Dict

def load_json_file(file_path: str) -> List[Dict]:
    """
    加载 JSON 文件并返回数据列表
    :param file_path: JSON 文件路径
    :return: 数据列表
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"文件 {file_path} 数据格式错误，非列表类型")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"文件 {file_path} 不存在，请检查路径")
    except json.JSONDecodeError:
        raise json.JSONDecodeError(f"文件 {file_path} 不是有效的 JSON 格式")

def extract_top_n_data(data: List[Dict], top_n: int = 100) -> List[Dict]:
    """
    提取前 N 条数据
    :param data: 原始数据列表
    :param top_n: 要提取的条数，默认100
    :return: 前 N 条数据
    """
    return data[:top_n]

def save_json_file(data: List[Dict], save_path: str) -> None:
    """
    保存数据到 JSON 文件
    :param data: 要保存的数据列表
    :param save_path: 保存路径
    """
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"成功保存 {len(data)} 条数据到 {save_path}")

if __name__ == "__main__":
    # 配置文件路径
    base_path = "/root/autodl-tmp/Medical-RAG/eval_results/base_results.json"
    rag_path = "/root/autodl-tmp/Medical-RAG/eval_results/rag_results.json"
    top_n = 100

    # 加载并提取 base_results 前100条
    base_data = load_json_file(base_path)
    base_top100 = extract_top_n_data(base_data, top_n)
    save_json_file(base_top100, "/root/autodl-tmp/Medical-RAG/eval_results/base_top100.json")

    # 加载并提取 rag_results 前100条
    rag_data = load_json_file(rag_path)
    rag_top100 = extract_top_n_data(rag_data, top_n)
    save_json_file(rag_top100, "/root/autodl-tmp/Medical-RAG/eval_results/rag_top100.json")

    # 数据校验
    print(f"\n数据校验结果：")
    print(f"base_results 原始条数: {len(base_data)} | 提取条数: {len(base_top100)}")
    print(f"rag_results 原始条数: {len(rag_data)} | 提取条数: {len(rag_top100)}")