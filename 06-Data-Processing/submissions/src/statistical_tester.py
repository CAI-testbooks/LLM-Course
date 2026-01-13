import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from rich.console import Console

console = Console()

class StatisticalTester:
    def __init__(self, llm_agent=None):
        self.llm_agent = llm_agent
    
    def perform_tests(self, data, output_dir):
        """执行统计检验"""
        console.print("[bold blue]📈 统计检验...[/bold blue]")
        
        results = {}
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        
        # 正态性检验
        normality_tests = {}
        for col in numerical_cols:
            if len(data[col].dropna()) > 3:
                stat, p = stats.shapiro(data[col].dropna())
                normality_tests[col] = {
                    'statistic': stat,
                    'p_value': p,
                    'normal': p > 0.05
                }
        
        results['normality'] = normality_tests
        
        # 相关性检验
        if len(numerical_cols) > 1:
            corr_matrix = data[numerical_cols].corr()
            results['correlation_matrix'] = corr_matrix
        
        console.print("[green]✅ 统计检验完成[/green]")
        return results