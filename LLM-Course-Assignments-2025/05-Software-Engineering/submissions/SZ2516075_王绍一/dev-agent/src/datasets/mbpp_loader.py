# src/datasets/mbpp_loader.py
from datasets import load_dataset

class MBPPLoader:
    """加载MBPP数据集"""
    
    def __init__(self, cache_dir: str = "./data/mbpp"):
        self.cache_dir = cache_dir
    
    def load(self, split: str = "train"):
        """加载数据集"""
        dataset = load_dataset("mbpp", split=split, cache_dir=self.cache_dir)
        return dataset
    
    def get_problem(self, idx: int) -> Dict:
        """获取指定索引的问题"""
        dataset = self.load()
        return dataset[idx] if idx < len(dataset) else None