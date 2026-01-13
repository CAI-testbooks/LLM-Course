"""
数据清洗模块
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import missingno as msno
from loguru import logger
from rich.console import Console
import matplotlib.pyplot as plt

console = Console()

class DataCleaner:
    """数据清洗器"""
    
    def __init__(self, llm_agent=None):
        self.llm_agent = llm_agent
        self.cleaning_strategies = {}
        
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        自动数据清洗
        
        Args:
            data: 原始数据
            
        Returns:
            清洗后的数据
        """
        console.print("[bold blue]🔧 开始数据清洗...[/bold blue]")
        
        original_shape = data.shape
        cleaned_data = data.copy()
        
        # 1. 处理缺失值
        cleaned_data = self._handle_missing_values(cleaned_data)
        
        # 2. 处理重复值
        cleaned_data = self._handle_duplicates(cleaned_data)
        
        # 3. 数据类型转换
        cleaned_data = self._convert_data_types(cleaned_data)
        
        # 4. 异常值检测与处理
        cleaned_data = self._handle_outliers(cleaned_data)
        
        # 5. 标准化列名
        cleaned_data = self._standardize_column_names(cleaned_data)
        
        # 6. 数据一致性检查
        cleaned_data = self._check_data_consistency(cleaned_data)
        
        final_shape = cleaned_data.shape
        
        console.print(f"[green]✅ 数据清洗完成[/green]")
        console.print(f"原始数据形状: {original_shape}")
        console.print(f"清洗后形状: {final_shape}")
        console.print(f"删除行数: {original_shape[0] - final_shape[0]}")
        console.print(f"删除列数: {original_shape[1] - final_shape[1]}")
        
        return cleaned_data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        console.print("📊 处理缺失值...")
        
        missing_summary = data.isnull().sum()
        missing_percentage = (missing_summary / len(data)) * 100
        
        if missing_summary.sum() > 0:
            # 可视化缺失值
            plt.figure(figsize=(12, 6))
            msno.matrix(data)
            plt.title('Missing Values Matrix')
            plt.savefig('results/figures/missing_values.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 使用LLM Agent决定处理策略
            if self.llm_agent:
                columns_with_missing = missing_percentage[missing_percentage > 0].index.tolist()
                data_description = f"数据集有{len(columns_with_missing)}列包含缺失值"
                
                response = self.llm_agent.analyze_data(
                    task_description="处理缺失值",
                    data_context=f"缺失值百分比:\n{missing_percentage[missing_percentage > 0].to_string()}",
                    analysis_type="missing_values"
                )
                
                if response:
                    logger.info(f"LLM建议: {response.get('insights', '')}")
            
            # 应用处理策略
            for col in data.columns:
                missing_pct = missing_percentage[col]
                
                if missing_pct == 0:
                    continue
                elif missing_pct < 5:
                    # 删除少量缺失的行
                    data = data.dropna(subset=[col])
                elif missing_pct < 30:
                    # 使用中位数/众数填充
                    if data[col].dtype in ['int64', 'float64']:
                        data[col] = data[col].fillna(data[col].median())
                    else:
                        data[col] = data[col].fillna(data[col].mode()[0])
                else:
                    # 考虑删除列或使用更复杂的方法
                    logger.warning(f"列 {col} 有 {missing_pct:.2f}% 的缺失值")
                    if missing_pct > 50:
                        data = data.drop(columns=[col])
        
        return data
    
    def _handle_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理重复值"""
        duplicates = data.duplicated().sum()
        
        if duplicates > 0:
            console.print(f"发现 {duplicates} 个重复行，正在删除...")
            data = data.drop_duplicates()
        
        return data
    
    def _convert_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """数据类型转换"""
        console.print("🔄 优化数据类型...")
        
        for col in data.columns:
            col_dtype = data[col].dtype
            
            # 尝试将数值字符串转换为数值类型
            if col_dtype == 'object':
                try:
                    data[col] = pd.to_numeric(data[col], errors='ignore')
                except:
                    pass
            
            # 优化内存使用
            if pd.api.types.is_integer_dtype(data[col]):
                data[col] = pd.to_numeric(data[col], downcast='integer')
            elif pd.api.types.is_float_dtype(data[col]):
                data[col] = pd.to_numeric(data[col], downcast='float')
        
        return data
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理异常值"""
        console.print("📈 检测异常值...")
        
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        
        outlier_report = {}
        
        for col in numerical_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            
            if len(outliers) > 0:
                outlier_percentage = (len(outliers) / len(data)) * 100
                outlier_report[col] = {
                    'count': len(outliers),
                    'percentage': outlier_percentage,
                    'method': 'winsorize' if outlier_percentage < 5 else 'keep'
                }
                
                # 处理异常值
                if outlier_percentage < 5:
                    # Winsorization
                    data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])
                    data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
        
        if outlier_report:
            console.print("📋 异常值报告:")
            for col, info in outlier_report.items():
                console.print(f"  {col}: {info['count']} 个异常值 ({info['percentage']:.2f}%) - 处理方法: {info['method']}")
        
        return data
    
    def _standardize_column_names(self, data: pd.DataFrame) -> pd.DataFrame:
        """标准化列名"""
        new_columns = {}
        
        for col in data.columns:
            # 移除特殊字符，转换为小写，用下划线替换空格
            new_name = col.strip().lower().replace(' ', '_').replace('-', '_')
            new_name = ''.join(e for e in new_name if e.isalnum() or e == '_')
            new_columns[col] = new_name
        
        data = data.rename(columns=new_columns)
        return data
    
    def _check_data_consistency(self, data: pd.DataFrame) -> pd.DataFrame:
        """数据一致性检查"""
        console.print("🔍 检查数据一致性...")
        
        # 检查是否有无限值
        inf_count = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            console.print(f"⚠️  发现 {inf_count} 个无限值")
            # 替换为NaN
            data = data.replace([np.inf, -np.inf], np.nan)
            data = self._handle_missing_values(data)
        
        return data
    
    def clean_data_interactive(self, data: pd.DataFrame) -> pd.DataFrame:
        """交互式数据清洗"""
        console.print("[bold yellow]🧹 交互式数据清洗[/bold yellow]")
        
        cleaned_data = data.copy()
        
        while True:
            console.print("\n[bold cyan]数据清洗选项:[/bold cyan]")
            console.print("1. 查看数据概览")
            console.print("2. 查看缺失值情况")
            console.print("3. 处理缺失值")
            console.print("4. 删除重复值")
            console.print("5. 检测异常值")
            console.print("6. 标准化列名")
            console.print("7. 完成清洗")
            
            choice = console.input("[bold cyan]请选择 (1-7): [/bold cyan]").strip()
            
            if choice == "1":
                console.print(f"数据形状: {cleaned_data.shape}")
                console.print(f"数据类型:\n{cleaned_data.dtypes}")
                console.print(f"前5行数据:\n{cleaned_data.head()}")
            
            elif choice == "2":
                missing = cleaned_data.isnull().sum()
                if missing.sum() > 0:
                    console.print("缺失值情况:")
                    console.print(missing[missing > 0])
                else:
                    console.print("[green]没有缺失值[/green]")
            
            elif choice == "3":
                strategy = console.input("[bold cyan]选择缺失值处理策略 (drop/median/mean/mode): [/bold cyan]").strip()
                cleaned_data = self._apply_missing_value_strategy(cleaned_data, strategy)
            
            elif choice == "4":
                duplicates = cleaned_data.duplicated().sum()
                if duplicates > 0:
                    confirm = console.input(f"发现 {duplicates} 个重复行，是否删除？(y/n): ").strip().lower()
                    if confirm == 'y':
                        cleaned_data = cleaned_data.drop_duplicates()
                        console.print(f"[green]已删除 {duplicates} 个重复行[/green]")
                else:
                    console.print("[green]没有重复值[/green]")
            
            elif choice == "5":
                self._detect_and_handle_outliers_interactive(cleaned_data)
            
            elif choice == "6":
                cleaned_data = self._standardize_column_names(cleaned_data)
                console.print("[green]列名已标准化[/green]")
            
            elif choice == "7":
                console.print("[green]数据清洗完成[/green]")
                break
            
            else:
                console.print("[red]无效选择[/red]")
        
        return cleaned_data
    
    def _apply_missing_value_strategy(self, data: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """应用缺失值处理策略"""
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        for col in data.columns:
            if data[col].isnull().sum() > 0:
                if strategy == 'drop':
                    data = data.dropna(subset=[col])
                elif strategy == 'median' and col in numerical_cols:
                    data[col] = data[col].fillna(data[col].median())
                elif strategy == 'mean' and col in numerical_cols:
                    data[col] = data[col].fillna(data[col].mean())
                elif strategy == 'mode':
                    if col in categorical_cols:
                        data[col] = data[col].fillna(data[col].mode()[0])
                    else:
                        data[col] = data[col].fillna(data[col].median())
        
        return data
    
    def _detect_and_handle_outliers_interactive(self, data: pd.DataFrame):
        """交互式异常值检测"""
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            
            if len(outliers) > 0:
                console.print(f"\n列 '{col}' 发现 {len(outliers)} 个异常值")
                console.print(f"异常值范围: [{lower_bound:.4f}, {upper_bound:.4f}]")
                
                action = console.input("如何处理？(keep/winsorize/drop/show): ").strip().lower()
                
                if action == 'winsorize':
                    data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])
                    data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
                    console.print(f"[green]已对列 '{col}' 进行Winsorization处理[/green]")
                elif action == 'drop':
                    data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
                    console.print(f"[green]已删除列 '{col}' 的异常值[/green]")
                elif action == 'show':
                    console.print(f"异常值样本:\n{outliers[[col]].head()}")