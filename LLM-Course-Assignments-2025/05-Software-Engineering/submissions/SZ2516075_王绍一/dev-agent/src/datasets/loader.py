# src/data/loader.py
from datasets import load_dataset

def load_mbpp_dataset():
    """
    加载MBPP数据集
    """
    dataset = load_dataset("mbpp")
    
    print("数据集结构：")
    print(f"训练集：{len(dataset['train'])} 条")
    print(f"验证集：{len(dataset['validation'])} 条")
    print(f"测试集：{len(dataset['test'])} 条")
    
    # 查看一条数据示例
    example = dataset['train'][0]
    print("\n示例数据：")
    print(f"任务ID: {example['task_id']}")
    print(f"描述: {example['text']}")
    print(f"代码: {example['code']}")
    print(f"测试: {example['test_list']}")
    
    return dataset

# 在main.py中测试
if __name__ == "__main__":
    dataset = load_mbpp_dataset()