# scripts/evaluate.py
from src.evaluator import Evaluator
from src.config import RAGConfig
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def main():
    evaluator = Evaluator()

    # 这里加载测试数据
    # test_data = load_test_data()

    # 进行评估
    # results = evaluator.evaluate_rag(predictions, references)
    # print(f"评估结果: {results}")

    print("评估脚本待实现...")


if __name__ == "__main__":
    main()
