import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner
from src.feature_engineer import FeatureEngineer
from src.eda_analyzer import EDAAnalyzer
from src.statistical_tester import StatisticalTester
from src.model_builder import ModelBuilder
from src.report_generator import ReportGenerator
from src.llm_agent import LLMAgent
from configs.settings import Config

console = Console()

class CWRUAnalysisAgent:
    """CWRU数据分析代理系统"""
    
    def __init__(self, config_path=None):
        """初始化分析代理"""
        self.config = Config(config_path)
        self.llm_agent = LLMAgent(self.config)
        
        # 初始化各模块
        self.data_loader = DataLoader()
        self.data_cleaner = DataCleaner(self.llm_agent)
        self.feature_engineer = FeatureEngineer(self.llm_agent)
        self.eda_analyzer = EDAAnalyzer(self.llm_agent)
        self.statistical_tester = StatisticalTester(self.llm_agent)
        self.model_builder = ModelBuilder(self.llm_agent)
        self.report_generator = ReportGenerator(self.llm_agent)
        
        self.data = None
        self.cleaned_data = None
        self.features = None
        self.results = {}
        
        logger.add(
            "logs/analysis.log",
            rotation="500 MB",
            retention="10 days",
            level="INFO"
        )
    
    def run_full_analysis(self, data_path, output_dir="results"):
        """
        运行完整的数据分析流程
        
        Args:
            data_path: 数据文件路径
            output_dir: 输出目录
        """
        console.print("[bold cyan]🚀 CWRU数据分析代理系统启动[/bold cyan]")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/figures", exist_ok=True)
        os.makedirs(f"{output_dir}/models", exist_ok=True)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            # 1. 数据加载
            task1 = progress.add_task("[blue]步骤1: 数据加载...", total=100)
            self.data = self.data_loader.load_data(data_path)
            progress.update(task1, completed=100)
            
            # 2. 数据清洗
            task2 = progress.add_task("[blue]步骤2: 数据清洗...", total=100)
            self.cleaned_data = self.data_cleaner.clean_data(self.data)
            progress.update(task2, completed=100)
            
            # 3. 特征工程
            task3 = progress.add_task("[blue]步骤3: 特征工程...", total=100)
            self.features = self.feature_engineer.create_features(self.cleaned_data)
            progress.update(task3, completed=100)
            
            # 4. EDA分析
            task4 = progress.add_task("[blue]步骤4: 探索性数据分析...", total=100)
            eda_results = self.eda_analyzer.analyze(self.features, output_dir)
            self.results['eda'] = eda_results
            progress.update(task4, completed=100)
            
            # 5. 统计检验
            task5 = progress.add_task("[blue]步骤5: 统计检验...", total=100)
            stat_results = self.statistical_tester.perform_tests(self.features, output_dir)
            self.results['statistics'] = stat_results
            progress.update(task5, completed=100)
            
            # 6. 建模分析
            task6 = progress.add_task("[blue]步骤6: 建模分析...", total=100)
            model_results = self.model_builder.build_models(self.features, output_dir)
            self.results['models'] = model_results
            progress.update(task6, completed=100)
            
            # 7. 生成报告
            task7 = progress.add_task("[blue]步骤7: 生成分析报告...", total=100)
            report_path = self.report_generator.generate_report(
                self.results, self.data, output_dir
            )
            progress.update(task7, completed=100)
        
        console.print(f"[bold green]✅ 分析完成！报告保存至: {report_path}[/bold green]")
        return report_path
    
    def interactive_analysis(self, data_path):
        """交互式数据分析"""
        console.print("[bold cyan]🤖 交互式数据分析模式[/bold cyan]")
        
        self.data = self.data_loader.load_data(data_path)
        
        while True:
            console.print("\n[bold yellow]请选择分析任务:[/bold yellow]")
            console.print("1. 数据概览")
            console.print("2. 数据清洗")
            console.print("3. 特征工程")
            console.print("4. 可视化分析")
            console.print("5. 统计检验")
            console.print("6. 机器学习建模")
            console.print("7. 生成完整报告")
            console.print("8. 退出")
            
            choice = console.input("[bold cyan]请输入选择 (1-8): [/bold cyan]").strip()
            
            if choice == "1":
                self.data_loader.summarize_data(self.data)
            elif choice == "2":
                self.cleaned_data = self.data_cleaner.clean_data_interactive(self.data)
            elif choice == "3":
                if self.cleaned_data is not None:
                    self.features = self.feature_engineer.create_features_interactive(self.cleaned_data)
                else:
                    console.print("[red]请先进行数据清洗！[/red]")
            elif choice == "4":
                if self.features is not None:
                    self.eda_analyzer.interactive_visualization(self.features)
                else:
                    console.print("[red]请先进行特征工程！[/red]")
            elif choice == "5":
                if self.features is not None:
                    self.statistical_tester.interactive_tests(self.features)
                else:
                    console.print("[red]请先进行特征工程！[/red]")
            elif choice == "6":
                if self.features is not None:
                    self.model_builder.interactive_modeling(self.features)
                else:
                    console.print("[red]请先进行特征工程！[/red]")
            elif choice == "7":
                report_path = self.run_full_analysis(data_path)
                console.print(f"[green]报告已生成: {report_path}[/green]")
                break
            elif choice == "8":
                console.print("[yellow]再见！[/yellow]")
                break
            else:
                console.print("[red]无效选择，请重试！[/red]")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CWRU数据分析代理系统")
    parser.add_argument("--data", type=str, required=True, help="数据文件路径")
    parser.add_argument("--mode", type=str, choices=["full", "interactive"], 
                       default="full", help="运行模式")
    parser.add_argument("--output", type=str, default="results", 
                       help="输出目录")
    parser.add_argument("--config", type=str, help="配置文件路径")
    
    args = parser.parse_args()
    
    # 初始化代理
    agent = CWRUAnalysisAgent(args.config)
    
    if args.mode == "full":
        agent.run_full_analysis(args.data, args.output)
    else:
        agent.interactive_analysis(args.data)

if __name__ == "__main__":
    main()