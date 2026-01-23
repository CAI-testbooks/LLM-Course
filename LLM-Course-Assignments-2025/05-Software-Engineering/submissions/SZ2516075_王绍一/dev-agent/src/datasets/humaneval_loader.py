# src/datasets/humaneval_loader.py
import json
from typing import List, Dict
from datasets import load_dataset
from typing import Dict, List, Any, Optional

class HumanEvalLoader:
    """加载HumanEval数据集"""
    
    def __init__(self, cache_dir: str = "./data/humaneval"):
        self.cache_dir = cache_dir
    
    def load(self) -> List[Dict]:
        """加载数据集"""
        try:
            dataset = load_dataset("openai_humaneval", cache_dir=self.cache_dir)
            return dataset["test"]
        except Exception as e:
            print(f"加载HumanEval失败: {e}")
            return self._load_fallback()
    
    def _load_fallback(self) -> List[Dict]:
        """备用加载方式"""
        import requests
        url = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl"
        response = requests.get(url)
        data = [json.loads(line) for line in response.text.strip().split('\n')]
        return data
    
    def get_task_by_id(self, task_id: str) -> Dict:
        """根据ID获取任务"""
        dataset = self.load()
        for item in dataset:
            if item.get("task_id") == task_id:
                return item
        return None