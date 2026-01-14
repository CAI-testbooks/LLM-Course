# scripts/train.py
from src.fine_tuner import FineTuner
from src.config import RAGConfig
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def main():
    config = RAGConfig.from_yaml("configs/config.yaml")

    fine_tuner = FineTuner(config)

    # 加载数据集
    train_dataset = fine_tuner.prepare_dataset(
        os.path.join(config.paths.fine_tune_data, "train"))
    eval_dataset = fine_tuner.prepare_dataset(
        os.path.join(config.paths.fine_tune_data, "eval"))

    # 开始训练
    trainer = fine_tuner.train(train_dataset, eval_dataset)
    print("训练完成!")


if __name__ == "__main__":
    main()
