import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from rich.console import Console

console = Console()

class ModelBuilder:
    def __init__(self, llm_agent=None):
        self.llm_agent = llm_agent
    
    def build_models(self, data, output_dir):
        """构建模型"""
        console.print("[bold blue]🤖 构建机器学习模型...[/bold blue]")
        
        results = {}
        
        # 假设最后一列是目标变量
        if len(data.columns) > 1:
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
            
            # 如果有足够的数据，尝试分类
            if y.nunique() > 1 and y.nunique() < len(y) * 0.5:
                # 数据分割
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # 标准化
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # 训练模型
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train_scaled, y_train)
                
                # 评估
                y_pred = model.predict(X_test_scaled)
                
                results['model'] = model
                results['scaler'] = scaler
                results['accuracy'] = model.score(X_test_scaled, y_test)
                results['classification_report'] = classification_report(y_test, y_pred)
                
                console.print(f"[green]✅ 模型训练完成[/green]")
                console.print(f"准确率: {results['accuracy']:.4f}")
        
        return results