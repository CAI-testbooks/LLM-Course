# src/datasets/dataset_manager.py
import os
from typing import Dict, List, Any
from datasets import load_dataset, Dataset

class DatasetManager:
    """管理所有代码基准数据集"""
    
    def __init__(self, cache_dir: str = "./data"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # 数据集映射
        self.dataset_configs = {
            "humaneval": {
                "name": "openai_humaneval",
                "split": "test",
                "type": "code_generation"
            },
            "mbpp": {
                "name": "mbpp",
                "split": "test",
                "type": "code_generation"
            },
            "swebench_lite": {
                "name": "princeton-nlp/SWE-bench_Lite",
                "split": "test",
                "type": "bug_fixing"
            }
        }
    
    def load_dataset(self, dataset_name: str) -> Dataset:
        """加载指定数据集"""
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"未知数据集: {dataset_name}")
        
        config = self.dataset_configs[dataset_name]
        print(f"正在加载数据集: {dataset_name}")
        
        try:
            dataset = load_dataset(
                config["name"],
                split=config["split"],
                cache_dir=os.path.join(self.cache_dir, dataset_name)
            )
            print(f"✅ 数据集加载成功: {len(dataset)} 条数据")
            return dataset
        except Exception as e:
            print(f"❌ 数据集加载失败: {e}")
            # 返回模拟数据
            return self._get_mock_dataset(dataset_name)
    
    def _get_mock_dataset(self, dataset_name: str) -> Dataset:
        """获取模拟数据集"""
        from datasets import Dataset
        
        if dataset_name == "humaneval":
            data = [
                {
                    "task_id": "HumanEval/0",
                    "prompt": "写一个函数，反转字符串",
                    "test": "assert reverse_string('hello') == 'olleh'\nassert reverse_string('') == ''",
                    "entry_point": "reverse_string"
                },
                {
                    "task_id": "HumanEval/1",
                    "prompt": "写一个函数，计算阶乘",
                    "test": "assert factorial(5) == 120\nassert factorial(0) == 1",
                    "entry_point": "factorial"
                }
            ]
        elif dataset_name == "mbpp":
            data = [
                {
                    "task_id": "mbpp/1",
                    "text": "写一个函数检查素数",
                    "code": "def is_prime(n):\n    if n <= 1:\n        return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0:\n            return False\n    return True",
                    "test_list": ["assert is_prime(17) == True", "assert is_prime(20) == False"]
                }
            ]
        else:  # swebench
            data = [
                {
                    "problem_id": "swe-bench/1",
                    "repo": "test_repo",
                    "instance_id": "1",
                    "base_commit": "abc123",
                    "test_patch": "test patch content",
                    "problem_statement": "修复字符串反转函数的bug",
                    "hints_text": "注意边界条件处理"
                }
            ]
        
        return Dataset.from_list(data)
    
    def get_sample(self, dataset_name: str, index: int = 0) -> Dict[str, Any]:
        """获取数据样本"""
        dataset = self.load_dataset(dataset_name)
        if 0 <= index < len(dataset):
            return dataset[index]
        return {}
    
    def get_all_datasets_info(self) -> Dict[str, Dict]:
        """获取所有数据集信息"""
        info = {}
        for name, config in self.dataset_configs.items():
            try:
                dataset = self.load_dataset(name)
                info[name] = {
                    "size": len(dataset),
                    "type": config["type"],
                    "loaded": True
                }
            except:
                info[name] = {
                    "size": 0,
                    "type": config["type"],
                    "loaded": False
                }
        return info