import pandas as pd
import numpy as np
from rich.console import Console

console = Console()

class DataLoader:
    def load_data(self, filepath):
        """加载数据"""
        console.print(f"[blue]📂 加载数据: {filepath}[/blue]")
        data = pd.read_csv(filepath)
        console.print(f"[green]✅ 数据加载成功[/green]")
        console.print(f"数据形状: {data.shape}")
        console.print(f"列名: {list(data.columns)}")
        return data
    
    def summarize_data(self, data):
        """数据概览"""
        console.print("[bold cyan]📊 数据概览[/bold cyan]")
        
        # 基本信息
        console.print(f"数据形状: {data.shape}")
        console.print(f"数据类型:\n{data.dtypes}")
        
        # 统计信息
        console.print(f"\n描述性统计:")
        console.print(data.describe())
        
        # 缺失值
        missing = data.isnull().sum()
        if missing.sum() > 0:
            console.print(f"\n缺失值情况:")
            console.print(missing[missing > 0])
        else:
            console.print(f"\n[green]没有缺失值[/green]")