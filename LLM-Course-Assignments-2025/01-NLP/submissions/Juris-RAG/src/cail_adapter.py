"""
数据处理兼容性适配脚本
自动选择使用精简文件(cail_cases_20k.json)或原始文件(cail_cases.json)
"""
import os
from pathlib import Path

def get_cail_file_path():
    """
    自动选择最优的CAIL数据文件
    优先使用精简版(20k条)，如果不存在则回退到原始文件
    """
    data_dir = Path("data/raw")
    
    # 优先级：精简版 > 原始版
    trimmed_file = data_dir / "cail_cases_20k.json"
    original_file = data_dir / "cail_cases.json"
    
    if trimmed_file.exists():
        size_mb = trimmed_file.stat().st_size / (1024**2)
        print(f"✓ 使用精简文件: cail_cases_20k.json ({size_mb:.2f} MB)")
        return str(trimmed_file)
    elif original_file.exists():
        size_gb = original_file.stat().st_size / (1024**3)
        print(f"⚠️ 原始文件: cail_cases.json ({size_gb:.2f} GB)")
        print(f"💡 建议运行 trim_cail_cases.py 精简数据")
        return str(original_file)
    else:
        raise FileNotFoundError(f"CAIL数据文件不存在: {data_dir}")


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
