import pandas as pd
import numpy as np
from datetime import datetime
from rich.console import Console

console = Console()

class ReportGenerator:
    def __init__(self, llm_agent=None):
        self.llm_agent = llm_agent
    
    def generate_report(self, results, original_data, output_dir):
        """生成报告"""
        console.print("[bold blue]📋 生成分析报告...[/bold blue]")
        
        report_path = f"{output_dir}/analysis_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# CWRU数据分析报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 1. 数据集概览\n\n")
            f.write(f"- **数据形状**: {original_data.shape}\n")
            f.write(f"- **特征数量**: {len(original_data.columns)}\n")
            f.write(f"- **数据类型**:\n")
            
            for col, dtype in original_data.dtypes.items():
                f.write(f"  - `{col}`: {dtype}\n")
            
            # EDA结果
            if 'eda' in results:
                f.write("\n## 2. 探索性数据分析\n\n")
                
                if 'summary' in results['eda']:
                    summary = results['eda']['summary']
                    f.write(f"- **缺失值总数**: {sum(original_data.isnull().sum().values)}\n")
                    
                    missing_cols = {k: v for k, v in summary.get('missing_percentage', {}).items() if v > 0}
                    if missing_cols:
                        f.write("\n**缺失值分布**:\n")
                        for col, pct in missing_cols.items():
                            f.write(f"  - `{col}`: {pct:.2f}% 缺失\n")
            
            # 统计检验结果
            if 'statistics' in results:
                f.write("\n## 3. 统计检验\n\n")
                
                if 'normality' in results['statistics']:
                    normality = results['statistics']['normality']
                    normal_cols = [col for col, test in normality.items() if test.get('normal', False)]
                    non_normal_cols = [col for col, test in normality.items() if not test.get('normal', False)]
                    
                    f.write(f"- **正态分布变量**: {len(normal_cols)} 个\n")
                    f.write(f"- **非正态分布变量**: {len(non_normal_cols)} 个\n")
            
            # 模型结果
            if 'models' in results:
                f.write("\n## 4. 建模分析\n\n")
                
                if 'accuracy' in results['models']:
                    f.write(f"- **模型准确率**: {results['models']['accuracy']:.4f}\n")
                
                if 'classification_report' in results['models']:
                    f.write("\n**分类报告**:\n```\n")
                    f.write(results['models']['classification_report'])
                    f.write("\n```\n")
            
            # 总结与建议
            f.write("\n## 5. 总结与建议\n\n")
            f.write("基于以上分析，建议：\n")
            f.write("1. 继续深入分析强相关特征\n")
            f.write("2. 考虑使用更复杂的模型\n")
            f.write("3. 进行特征重要性分析\n")
            f.write("4. 验证模型的泛化能力\n")
        
        console.print(f"[green]✅ 报告生成完成: {report_path}[/green]")
        return report_path