# run_project.py
import subprocess
import sys
import os
from pathlib import Path


def run_script():
    # 添加项目根目录到Python路径
    project_root = str(Path(__file__).parent)
    if project_root not in sys.path:
        sys.path.append(project_root)

    print("=== 开始故障诊断项目 ===")

    # 1. 转换CSV数据（如果需要）
    print("\n1. 检查数据格式...")
    try:
        # 直接使用CSV数据，不需要转换
        from config import opt
        print(f"将直接使用CSV文件进行训练")
        print(f"训练文件: {opt.train_data_root}")
        print(f"测试文件: {opt.test_data_root}")

    except Exception as e:
        print(f"配置错误: {e}")
        return

    # 2. 训练模型
    print("\n2. 开始训练模型...")
    try:
        from main import train
        train()
        print("训练完成！")
    except Exception as e:
        print(f"训练错误: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. 测试模型
    print("\n3. 开始测试模型...")
    try:
        from main import test
        test()
        print("测试完成！")
    except Exception as e:
        print(f"测试错误: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n=== 项目运行完成 ===")
    print("结果文件:")
    print("- 混淆矩阵: results/confusion_matrix.xlsx")
    print("- 特征文件: results/features.h5")
    print("- 模型文件: results/SimpleCNN_model.pth")


if __name__ == "__main__":
    run_script()