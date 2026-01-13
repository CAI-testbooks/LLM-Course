"""
探索性数据分析(EDA)模块
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Tuple
from rich.console import Console
from rich.table import Table
import warnings
warnings.filterwarnings('ignore')

console = Console()

class EDAAnalyzer:
    """EDA分析器"""
    
    def __init__(self, llm_agent=None):
        self.llm_agent = llm_agent
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def analyze(self, data: pd.DataFrame, output_dir: str = "results") -> Dict[str, Any]:
        """
        执行完整的EDA分析
        
        Args:
            data: 输入数据
            output_dir: 输出目录
            
        Returns:
            EDA分析结果
        """
        console.print("[bold blue]🔍 开始探索性数据分析...[/bold blue]")
        
        results = {
            'summary': {},
            'distributions': {},
            'correlations': {},
            'patterns': {},
            'insights': []
        }
        
        # 1. 数据概览
        results['summary'] = self._get_data_summary(data)
        
        # 2. 单变量分析
        console.print("📊 单变量分析...")
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        for col in numerical_cols:
            results['distributions'][col] = self._analyze_numerical_column(data, col, output_dir)
        
        for col in categorical_cols:
            results['distributions'][col] = self._analyze_categorical_column(data, col, output_dir)
        
        # 3. 双变量分析
        console.print("📈 双变量分析...")
        if len(numerical_cols) > 1:
            results['correlations'] = self._analyze_correlations(data, numerical_cols, output_dir)
        
        # 4. 多变量分析
        console.print("🌐 多变量分析...")
        if len(numerical_cols) >= 3:
            results['patterns'] = self._analyze_multivariate_patterns(data, numerical_cols, output_dir)
        
        # 5. 使用LLM生成洞察
        if self.llm_agent:
            console.print("🤖 使用LLM生成数据洞察...")
            insights = self._generate_insights_with_llm(data, results)
            results['insights'] = insights
        
        # 6. 生成EDA报告
        self._generate_eda_report(results, output_dir)
        
        console.print("[green]✅ EDA分析完成[/green]")
        return results
    
    def _get_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """获取数据摘要"""
        summary = {
            'shape': data.shape,
            'dtypes': data.dtypes.to_dict(),
            'memory_usage': data.memory_usage(deep=True).sum(),
            'missing_values': data.isnull().sum().to_dict(),
            'missing_percentage': (data.isnull().sum() / len(data) * 100).to_dict(),
            'numerical_stats': {},
            'categorical_stats': {}
        }
        
        # 数值型数据统计
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            summary['numerical_stats'] = data[numerical_cols].describe().to_dict()
        
        # 分类型数据统计
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                summary['categorical_stats'][col] = {
                    'unique_count': data[col].nunique(),
                    'top_values': data[col].value_counts().head(5).to_dict()
                }
        
        return summary
    
    def _analyze_numerical_column(self, data: pd.DataFrame, column: str, 
                                output_dir: str) -> Dict[str, Any]:
        """分析数值型列"""
        results = {}
        
        # 基本统计
        stats = data[column].describe().to_dict()
        stats['skewness'] = data[column].skew()
        stats['kurtosis'] = data[column].kurtosis()
        stats['cv'] = data[column].std() / data[column].mean() if data[column].mean() != 0 else np.nan
        
        results['statistics'] = stats
        
        # 可视化
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 直方图
        axes[0, 0].hist(data[column].dropna(), bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title(f'{column} - Distribution', fontsize=12)
        axes[0, 0].set_xlabel(column)
        axes[0, 0].set_ylabel('Frequency')
        
        # 箱线图
        axes[0, 1].boxplot(data[column].dropna())
        axes[0, 1].set_title(f'{column} - Box Plot', fontsize=12)
        axes[0, 1].set_ylabel(column)
        
        # Q-Q图
        from scipy import stats
        stats.probplot(data[column].dropna(), dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title(f'{column} - Q-Q Plot', fontsize=12)
        
        # 核密度估计
        sns.kdeplot(data[column].dropna(), ax=axes[1, 1], fill=True)
        axes[1, 1].set_title(f'{column} - KDE Plot', fontsize=12)
        axes[1, 1].set_xlabel(column)
        
        plt.suptitle(f'Analysis of {column}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/figures/{column}_distribution.png', dpi=300)
        plt.close()
        
        # 交互式图表 (Plotly)
        fig = px.histogram(data, x=column, marginal='box', 
                         title=f'Distribution of {column}')
        fig.write_html(f'{output_dir}/figures/{column}_interactive.html')
        
        return results
    
    def _analyze_categorical_column(self, data: pd.DataFrame, column: str, 
                                  output_dir: str) -> Dict[str, Any]:
        """分析分类型列"""
        results = {}
        
        value_counts = data[column].value_counts()
        results['value_counts'] = value_counts.to_dict()
        results['unique_count'] = data[column].nunique()
        results['top_value'] = value_counts.index[0] if len(value_counts) > 0 else None
        results['top_percentage'] = (value_counts.iloc[0] / len(data) * 100) if len(value_counts) > 0 else 0
        
        # 可视化
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 条形图
        top_n = min(20, len(value_counts))
        axes[0].barh(range(top_n), value_counts.head(top_n).values)
        axes[0].set_yticks(range(top_n))
        axes[0].set_yticklabels(value_counts.head(top_n).index)
        axes[0].set_title(f'{column} - Top {top_n} Categories', fontsize=12)
        axes[0].set_xlabel('Count')
        
        # 饼图 (仅显示前5个类别)
        top_5 = value_counts.head(5)
        other_sum = value_counts[5:].sum() if len(value_counts) > 5 else 0
        
        if other_sum > 0:
            top_5['Other'] = other_sum
        
        axes[1].pie(top_5.values, labels=top_5.index, autopct='%1.1f%%')
        axes[1].set_title(f'{column} - Distribution', fontsize=12)
        
        plt.suptitle(f'Analysis of {column}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/figures/{column}_categorical.png', dpi=300)
        plt.close()
        
        return results
    
    def _analyze_correlations(self, data: pd.DataFrame, numerical_cols: List[str], 
                            output_dir: str) -> Dict[str, Any]:
        """分析相关性"""
        results = {}
        
        # 计算相关系数矩阵
        corr_matrix = data[numerical_cols].corr()
        results['correlation_matrix'] = corr_matrix.to_dict()
        
        # 找出强相关性
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    strong_correlations.append({
                        'pair': (corr_matrix.columns[i], corr_matrix.columns[j]),
                        'correlation': corr_value
                    })
        
        results['strong_correlations'] = strong_correlations
        
        # 可视化
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True, linewidths=0.5)
        plt.title('Correlation Matrix', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/figures/correlation_matrix.png', dpi=300)
        plt.close()
        
        # 散点图矩阵
        if len(numerical_cols) <= 8:  # 避免图太大
            scatter_matrix = pd.plotting.scatter_matrix(
                data[numerical_cols], 
                figsize=(15, 15),
                diagonal='kde',
                alpha=0.5
            )
            plt.suptitle('Scatter Matrix of Numerical Variables', fontsize=16)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/figures/scatter_matrix.png', dpi=300)
            plt.close()
        
        return results
    
    def _analyze_multivariate_patterns(self, data: pd.DataFrame, 
                                     numerical_cols: List[str], 
                                     output_dir: str) -> Dict[str, Any]:
        """分析多变量模式"""
        results = {}
        
        # PCA分析
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # 标准化数据
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[numerical_cols].fillna(0))
        
        # 执行PCA
        pca = PCA()
        pca_result = pca.fit_transform(scaled_data)
        
        results['pca'] = {
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'components': pca.components_.tolist()
        }
        
        # 可视化PCA结果
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 解释方差比
        axes[0].plot(range(1, len(pca.explained_variance_ratio_) + 1), 
                    pca.explained_variance_ratio_, 'bo-')
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Explained Variance Ratio')
        axes[0].set_title('Scree Plot')
        axes[0].grid(True)
        
        # 累积解释方差
        axes[1].plot(range(1, len(pca.explained_variance_ratio_) + 1), 
                    np.cumsum(pca.explained_variance_ratio_), 'ro-')
        axes[1].set_xlabel('Number of Components')
        axes[1].set_ylabel('Cumulative Explained Variance')
        axes[1].set_title('Cumulative Explained Variance')
        axes[1].grid(True)
        
        plt.suptitle('PCA Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/figures/pca_analysis.png', dpi=300)
        plt.close()
        
        return results
    
    def _generate_insights_with_llm(self, data: pd.DataFrame, 
                                  eda_results: Dict[str, Any]) -> List[str]:
        """使用LLM生成洞察"""
        if not self.llm_agent:
            return []
        
        # 准备数据摘要
        data_summary = f"""
        数据集信息:
        - 形状: {eda_results['summary']['shape']}
        - 数值列数: {len(data.select_dtypes(include=[np.number]).columns)}
        - 分类列数: {len(data.select_dtypes(include=['object', 'category']).columns)}
        - 缺失值比例: {data.isnull().sum().sum() / data.size:.2%}
        
        关键发现:
        - 强相关性: {len(eda_results.get('correlations', {}).get('strong_correlations', []))} 对
        - 偏态分布: {sum([1 for col, stats in eda_results['distributions'].items() 
                         if 'statistics' in stats and abs(stats['statistics'].get('skewness', 0)) > 1])}
        """
        
        # 获取LLM洞察
        response = self.llm_agent.analyze_data(
            task_description="基于EDA结果生成数据洞察和建议",
            data_context=data_summary,
            analysis_type="eda_insights"
        )
        
        if response and 'insights' in response:
            insights = response['insights'].split('\n')
            return [insight.strip() for insight in insights if insight.strip()]
        
        return []
    
    def _generate_eda_report(self, results: Dict[str, Any], output_dir: str):
        """生成EDA报告"""
        report_path = f"{output_dir}/eda_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# EDA分析报告\n\n")
            
            # 数据概览
            f.write("## 1. 数据概览\n\n")
            f.write(f"- **数据形状**: {results['summary']['shape']}\n")
            f.write(f"- **内存使用**: {results['summary']['memory_usage'] / 1024**2:.2f} MB\n")
            
            # 缺失值分析
            missing_info = results['summary']['missing_percentage']
            missing_cols = {k: v for k, v in missing_info.items() if v > 0}
            
            if missing_cols:
                f.write("\n## 2. 缺失值分析\n\n")
                f.write("| 列名 | 缺失百分比 |\n")
                f.write("|------|------------|\n")
                for col, pct in missing_cols.items():
                    f.write(f"| {col} | {pct:.2f}% |\n")
            
            # 分布分析
            f.write("\n## 3. 变量分布\n\n")
            
            for col, stats in results['distributions'].items():
                if 'statistics' in stats:  # 数值型变量
                    stat_info = stats['statistics']
                    f.write(f"### {col}\n")
                    f.write(f"- **偏度**: {stat_info.get('skewness', 'N/A'):.4f}\n")
                    f.write(f"- **峰度**: {stat_info.get('kurtosis', 'N/A'):.4f}\n")
                    f.write(f"- **变异系数**: {stat_info.get('cv', 'N/A'):.4f}\n\n")
                elif 'value_counts' in stats:  # 分类型变量
                    f.write(f"### {col}\n")
                    f.write(f"- **唯一值数量**: {stats['unique_count']}\n")
                    f.write(f"- **最常见值**: {stats['top_value']} ({stats['top_percentage']:.2f}%)\n\n")
            
            # 相关性分析
            if 'correlations' in results:
                f.write("\n## 4. 相关性分析\n\n")
                
                strong_corrs = results['correlations'].get('strong_correlations', [])
                if strong_corrs:
                    f.write("### 强相关性对 (|r| > 0.7)\n\n")
                    f.write("| 变量对 | 相关系数 |\n")
                    f.write("|--------|----------|\n")
                    for corr in strong_corrs:
                        pair = corr['pair']
                        f.write(f"| {pair[0]} - {pair[1]} | {corr['correlation']:.4f} |\n")
            
            # LLM洞察
            if results['insights']:
                f.write("\n## 5. AI洞察与建议\n\n")
                for i, insight in enumerate(results['insights'], 1):
                    f.write(f"{i}. {insight}\n")
            
            # 可视化文件列表
            f.write("\n## 6. 生成的可视化文件\n\n")
            import os
            figure_dir = f"{output_dir}/figures"
            if os.path.exists(figure_dir):
                figures = [f for f in os.listdir(figure_dir) if f.endswith(('.png', '.html'))]
                for figure in figures:
                    f.write(f"- `{figure_dir}/{figure}`\n")
        
        console.print(f"[green]📄 EDA报告已保存至: {report_path}[/green]")
    
    def interactive_visualization(self, data: pd.DataFrame):
        """交互式可视化"""
        console.print("[bold yellow]🎨 交互式可视化[/bold yellow]")
        
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        while True:
            console.print("\n[bold cyan]可视化选项:[/bold cyan]")
            console.print("1. 单变量分布")
            console.print("2. 双变量关系")
            console.print("3. 相关性热图")
            console.print("4. 时间序列分析")
            console.print("5. 多变量分析")
            console.print("6. 自定义Plotly图表")
            console.print("7. 返回")
            
            choice = console.input("[bold cyan]请选择 (1-7): [/bold cyan]").strip()
            
            if choice == "1":
                self._univariate_visualization_interactive(data, numerical_cols, categorical_cols)
            elif choice == "2":
                self._bivariate_visualization_interactive(data, numerical_cols, categorical_cols)
            elif choice == "3":
                self._plot_correlation_matrix_interactive(data, numerical_cols)
            elif choice == "4":
                self._time_series_analysis_interactive(data)
            elif choice == "5":
                self._multivariate_analysis_interactive(data, numerical_cols)
            elif choice == "6":
                self._custom_plotly_visualization_interactive(data)
            elif choice == "7":
                break
            else:
                console.print("[red]无效选择[/red]")
    
    def _univariate_visualization_interactive(self, data, numerical_cols, categorical_cols):
        """交互式单变量可视化"""
        console.print("\n[bold]选择变量类型:[/bold]")
        console.print("1. 数值变量")
        console.print("2. 分类变量")
        
        var_type = console.input("选择 (1-2): ").strip()
        
        if var_type == "1":
            console.print(f"可用数值变量: {numerical_cols}")
            selected = console.input("选择变量 (用逗号分隔): ").strip().split(',')
            selected = [col.strip() for col in selected if col.strip() in numerical_cols]
            
            for col in selected:
                self._plot_univariate_numerical(data, col)
        
        elif var_type == "2":
            if categorical_cols:
                console.print(f"可用分类变量: {categorical_cols}")
                selected = console.input("选择变量: ").strip()
                if selected in categorical_cols:
                    self._plot_univariate_categorical(data, selected)
            else:
                console.print("[red]没有分类变量[/red]")
    
    def _plot_univariate_numerical(self, data, column):
        """绘制数值变量单变量图"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('直方图', '箱线图', 'Q-Q图', '核密度估计'),
            specs=[[{'type': 'xy'}, {'type': 'xy'}],
                   [{'type': 'xy'}, {'type': 'xy'}]]
        )
        
        # 直方图
        fig.add_trace(
            go.Histogram(x=data[column].dropna(), name='Histogram'),
            row=1, col=1
        )
        
        # 箱线图
        fig.add_trace(
            go.Box(y=data[column].dropna(), name='Box Plot'),
            row=1, col=2
        )
        
        # Q-Q图
        from scipy import stats
        qq = stats.probplot(data[column].dropna(), dist="norm")
        x = qq[0][0]
        y = qq[0][1]
        
        fig.add_trace(
            go.Scatter(x=x, y=y, mode='markers', name='Q-Q Plot'),
            row=2, col=1
        )
        
        # 添加参考线
        fig.add_trace(
            go.Scatter(x=[x.min(), x.max()], y=[x.min(), x.max()], 
                      mode='lines', name='Normal Line', line=dict(dash='dash')),
            row=2, col=1
        )
        
        # 核密度估计
        import plotly.figure_factory as ff
        hist_data = [data[column].dropna()]
        group_labels = [column]
        
        fig_hist = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)
        
        for trace in fig_hist.data:
            fig.add_trace(trace, row=2, col=2)
        
        fig.update_layout(height=800, title_text=f"Analysis of {column}")
        fig.show()