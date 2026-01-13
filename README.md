## CWRU轴承故障诊断数据分析代理系统

# 一、程序概述
   本系统是一款以Case Western Reserve University美国西储大学(CWRU)轴承故障诊断数据集设计的自动化数据分析平台。系统采用模块化架构，结合传统数据分析方法与基于DSPy的大语言模型智能分析，提供从数据加载、清洗、特征工程到EDA分析、统计检验、建模预测和报告生成的完整自动化工作流。适用于工业设备故障诊断、数据分析作业、业务数据洞察等场景。

# 二、系统功能模块介绍

## 1. 核心模块
| 功能模块                 | 详细说明                                                               |
| ------------------------ | ----------------------------------------------------------------------- |
| 多源数据接入             | 支持CWRU轴承故障诊断数据集的CSV格式加载，兼容常见结构化数据格式         |
| 自动化数据清洗           | 自动化处理缺失值、重复值、异常值检测与处理，支持数据标准化和类型转换     |
| 智能特征工程             | 自动生成统计特征、分箱特征、标准化特征，支持特征选择建议               |
| 探索性数据分析(EDA)      | 自动计算数据描述性统计、变量分布特征、相关性分析，生成专业可视化图表   |
| 统计检验                 | 自动化执行正态性检验、相关性分析、显著性检验等统计分析                 |
| 机器学习建模             | 自动构建和评估机器学习模型，支持分类任务，输出模型性能报告             |
| LLM驱动的智能分析        | 基于DSPy框架的智能分析代理，提供数据洞察、建议和代码生成               |
| 专业报告生成             | 自动化生成结构化Markdown格式分析报告，支持可视化图表嵌入               |

## 2. 应用的可视化系统

-**多图表类型** ：直方图、箱线图、相关性热图、PCA分析图等

-**交互式图表** ：基于Plotly的交互式HTML图表

-**专业报告** ：自动生成包含可视化引用的Markdown报告

## 3. 灵活配置系统

-**YAML** : 配置文件：支持结构化配置管理

-**环境变量集成** ：通过.env文件管理敏感信息

-**多LLM支持** ：支持OpenAI GPT系列和Anthropic Claude模型

## 4. 智能数据分析代理

-**DSPy签名系统** ：通过声明式签名定义分析任务，如DataAnalysisSignature、FeatureEngineeringSignature

-**链式思考（Chain of Thought）** ：通过dspy.ChainOfThought实现复杂推理过程

-**智能优化** ：使用BootstrapFewShot优化提示词和示例选择

## 5. 自动化工作流

-**完整分析流程** ：7步自动化流程，涵盖数据科学全生命周期

-**交互式模式** ：支持交互式探索，用户可逐步执行分析任务

# 三、技术应用说明

## 1. 核心框架
| 技术组件       | 版本     | 作用                                                      |
| -------------- | -------- | --------------------------------------------------------- |
| DSPy           | 2.2.6    | 声明式语言模型编程框架，构建智能分析代理                    |
| OpenAI         | 1.12.0   | GPT系列模型API接口，提供智能分析能力                       |
| Anthropic      | 0.25.1   | Claude模型API接口（可选），提供备选LLM能力                 |
| Python         | 3.11.0   | 编程语言基础环境                                          |
## 2. 数据处理
| 技术组件       | 版本        | 作用                                                       |
| -------------- | ----------- | ---------------------------------------------------------- |
| Pandas         | 2.2.0       | 核心数据结构（DataFrame）、数据清洗、EDA计算                   |
| NumPy          | 1.26.4      | 数值计算、数组处理、特征工程                                  |
| Scikit-learn   | 1.4.1.post1 | 机器学习算法库、模型构建与评估                                |
| SciPy          | 1.12.0      | 统计检验（相关性分析、显著性检验）                            |
| StatsModels    | 0.14.1      | 拓展统计分析能力                                             |


## 3. 可视化
| 技术组件       | 版本    | 作用                                                       |
| -------------- | ------- | ---------------------------------------------------------- |
| Matplotlib     | 3.8.3   | 基础绘图引擎，生成直方图、箱线图、散点图等                     |
| Seaborn        | 0.13.2  | 美化可视化图表，简化分类变量/相关性可视化                      |
| Plotly         | 5.19.0  | 交互式可视化图表，支持HTML格式导出                            |
| Missingno      | 0.5.2   | 缺失值可视化分析                                             |


## 4. 辅助工具
| 技术组件       | 版本    | 作用                                                       |
| -------------- | ------- | ---------------------------------------------------------- |
| Rich           | 13.7.0  | 终端美化与进度显示，提升用户体验                              |
| Loguru         | 0.7.2   | 结构化日志记录，便于调试和监控                                |
| Python-dotenv  | 1.0.1   | 环境变量管理，安全存储API密钥                                 |
# 四、程序架构

 ```bash
 CWRU数据分析代理系统架构：
 ├── 输入层 (Input Layer)
 │   ├── 数据文件 (CSV格式)
 │   └── 配置文件 (YAML格式)
 ├── 处理层 (Processing Layer)
 │   ├── 数据加载模块 (DataLoader)
 │   ├── 数据清洗模块 (DataCleaner)
 │   ├── 特征工程模块 (FeatureEngineer)
 │   ├── EDA分析模块 (EDAAnalyzer)
 │   ├── 统计检验模块 (StatisticalTester)
 │   └── 建模模块 (ModelBuilder)
 ├── 智能层 (Intelligence Layer)
 │   └── LLM代理模块 (LLMAgent)
 ├── 输出层 (Output Layer)
 │   ├── 可视化图表 (PNG, HTML格式)
 │   ├── 分析报告 (Markdown格式)
 │   └── 模型文件 (pickle格式)
 └── 控制层 (Control Layer)
     └── 主控制器 (CWRUAnalysisAgent)
```

# 五、系统运行前的步骤准备

## 1. 环境准备
  请在Python 3.11+环境下进行如下所示的安装

```bash
  # 创建虚拟环境
   python -m venv venv

  # Windows激活
   venv\Scripts\activate.bat

  # Mac/Linux激活
   source venv/bin/activate
```

## 2. 安装所需库
  将以下内容保存为requirements.txt ，执行安装命令：
  
  ```txt
  # 核心依赖
  python==3.11.0
  dspy==2.2.6
  openai==1.12.0
  anthropic==0.25.1

  # 数据分析工具
  pandas==2.2.0
  numpy==1.26.4
  scikit-learn==1.4.1.post1
  statsmodels==0.14.1
  scipy==1.12.0

  # 可视化工具
  matplotlib==3.8.3
  seaborn==0.13.2
  plotly==5.19.0

  # 数据处理工具
  scikit-learn-intelex==2024.0.1
  missingno==0.5.2
  pycaret==3.3.0

  # 报告生成工具
  jupyter==1.0.0
  notebook==7.1.0
  nbconvert==7.14.2
  pandas-profiling==3.6.6
  markdown==3.5.2
  reportlab==4.1.0

  # 其他工具
  python-dotenv==1.0.1
  tqdm==4.66.2
  loguru==0.7.2
  rich==13.7.0
  ```
  
  ```bash
     # 进行所有工具及库的安装
     pip install -r requirements.txt 
  ```

# 六、快速开始指南

## 1. 环境配置

```bash
# 克隆项目
git clone <repository-url>
cd cwru-data-analysis-agent

# 创建虚拟环境（Windows）
python -m venv venv
venv\Scripts\activate

# 创建虚拟环境（Linux/Mac）
python -m venv venv
source venv/bin/activate
```

## 2. API密钥配置

创建.env文件：
```bash
env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## 3. 配置文件设置

编辑config.yaml：
```yaml
llm:
  provider: "openai"
  model: "gpt-4-turbo-preview"
  temperature: 0.1
  max_tokens: 2000
```

## 4. 运行分析
```bash
# 完整分析模式
python run_analysis.py --data data_12k_10c.csv --mode full --output results

# 交互式分析模式
python run_analysis.py --data data_12k_10c.csv --mode interactive
```

# 七、项目目录结构
```text
cwru-data-analysis-agent/
├── src/                          # 源代码目录
│   ├── main.py                   # 主控制器
│   ├── data_loader.py           # 数据加载模块
│   ├── data_cleaner.py          # 数据清洗模块
│   ├── feature_engineer.py      # 特征工程模块
│   ├── eda_analyzer.py          # EDA分析模块
│   ├── statistical_tester.py    # 统计检验模块
│   ├── model_builder.py         # 建模模块
│   ├── report_generator.py      # 报告生成模块
│   └── llm_agent.py             # LLM智能代理
├── configs/                      # 配置文件目录
│   └── settings.py              # 配置类定义
├── data/                         # 数据文件目录
│   └── data_12k_10c.csv         # 示例数据文件
├── results/                      # 输出结果目录
│   ├── figures/                 # 可视化图表
│   ├── models/                  # 保存的模型
│   └── analysis_report.md       # 分析报告
├── logs/                         # 日志文件目录
│   └── analysis.log             # 分析日志
├── tests/                        # 测试文件目录
├── .env                          # 环境变量文件
├── config.yaml                   # 配置文件
├── requirements.txt              # 依赖列表
├── run_analysis.py               # 运行脚本
├── setup.sh                      # 安装脚本
└── README.md                     # 项目说明文档
```

# 八、使用示例
## 1. 数据分析流程示例
```python
# 初始化分析代理
agent = CWRUAnalysisAgent("config.yaml")

# 执行完整分析
report_path = agent.run_full_analysis(
    data_path="data_12k_10c.csv",
    output_dir="results"
)

print(f"分析报告已生成: {report_path}")
```

## 2. 交互式分析示例
```python
  # 启动交互式分析
agent.interactive_analysis("data_12k_10c.csv")

# 交互式菜单将提供以下选项：
# 1. 数据概览
# 2. 数据清洗
# 3. 特征工程
# 4. 可视化分析
# 5. 统计检验
# 6. 机器学习建模
# 7. 生成完整报告
# 8. 退出
```

# 九、扩展与定制

## 1. 添加新分析模块
```python

# 1. 在src目录下创建新模块文件
# 2. 实现核心分析功能
# 3. 在主控制器中注册模块
# 4. 更新配置文件和运行脚本
```

## 2. 自定义LLM签名
```python
class CustomAnalysisSignature(dspy.Signature):
    """自定义分析签名"""
    input_data = dspy.InputField(desc="输入数据描述")
    analysis_params = dspy.InputField(desc="分析参数", optional=True)
    
    analysis_results = dspy.OutputField(desc="分析结果")
    recommendations = dspy.OutputField(desc="建议")    
# 在LLMAgent中注册使用
self.custom_analyzer = dspy.ChainOfThought(CustomAnalysisSignature)
```

## 3. 支持新数据格式
```python
class DataLoader:
    def load_data(self, filepath):
        """扩展支持多种数据格式"""
        if filepath.endswith('.csv'):
            return pd.read_csv(filepath)
        elif filepath.endswith('.xlsx'):
            return pd.read_excel(filepath)
        elif filepath.endswith('.json'):
            return pd.read_json(filepath)
        else:
            raise ValueError(f"不支持的文件格式: {filepath}")
```

# 十、性能与优化

## 1. 性能特征

数据处理：支持百万级数据行的处理

内存管理：自动优化数据类型减少内存占用

并行处理：可扩展支持多进程/多线程处理

## 2. 优化建议

对于大型数据集，建议分批次处理

使用特征选择减少不必要特征

调整LLM模型的temperature参数控制输出稳定性

# 十一、故障排除

常见问题及解决方案：

（1）DSPy API版本问题

症状：AttributeError: module 'dspy' has no attribute 'OpenAI'

解决：更新DSPy到2.2.6+版本，使用新的API配置方式

（2）OpenAI API密钥问题

症状：API调用失败或超时

解决：检查.env文件配置，确保API密钥有效

（3）内存不足问题

症状：处理大型数据集时内存溢出

解决：启用数据分块处理，优化数据类型

（4）可视化问题

症状：图表无法保存或显示

解决：确保输出目录存在且有写入权限
