import pandas as pd
import numpy as np
from rich.console import Console

console = Console()

class FeatureEngineer:
    def __init__(self, llm_agent=None):
        self.llm_agent = llm_agent
    
    def create_features(self, data):
        """创建特征"""
        console.print("[bold blue]🔧 特征工程...[/bold blue]")
        
        features = data.copy()
        
        # 数值特征的统计特征
        numerical_cols = features.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            # 添加统计特征
            features[f'{col}_zscore'] = (features[col] - features[col].mean()) / features[col].std()
            features[f'{col}_rank'] = features[col].rank()
            
            # 添加分箱特征
            if features[col].nunique() > 10:
                features[f'{col}_bin'] = pd.qcut(features[col], q=5, labels=False)
        
        console.print(f"[green]✅ 特征工程完成[/green]")
        console.print(f"原始特征数: {len(data.columns)}")
        console.print(f"新特征数: {len(features.columns)}")
        
        return features