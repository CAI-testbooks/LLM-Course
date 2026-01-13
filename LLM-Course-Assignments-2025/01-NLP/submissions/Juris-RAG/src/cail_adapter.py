"""
数据处理兼容性适配脚本
加载 CAIL 案例数据文件 (cail_cases.json)
"""
import os
from pathlib import Path

def get_cail_file_path():
    """
    返回 CAIL 案例文件路径
    """
    data_dir = Path("data/raw")
    
    cail_file = data_dir / "cail_cases.json"
    
    if cail_file.exists():
        size_mb = cail_file.stat().st_size / (1024**2)
        print(f"✓ 使用 CAIL 案例文件: cail_cases.json ({size_mb:.2f} MB)")
        return str(cail_file)
    else:
        print(f"⚠️ 未找到 CAIL 案例文件: cail_cases.json")
        return str(cail_file)


def recommend_case_limit():
    """根据文件大小推荐合适的案例加载数量"""
    cail_file = get_cail_file_path()
    size_mb = os.path.getsize(cail_file) / (1024**2)
    
    if size_mb < 50:  # 精简版
        return 20000
    else:  # 原始版，建议限制
        return 5000


if __name__ == "__main__":
    try:
        cail_file = get_cail_file_path()
        recommended_limit = recommend_case_limit()
        print(f"推荐加载案例数: {recommended_limit}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
