# CWRU数据集分析系统

一、项目概述
本系统是一个专为Case Western Reserve University美国西储大学(CWRU)轴承故障诊断数据集设计的自动化数据分析平台。系统采用模块化架构，结合传统数据驱动分析方法与基于DSPy的大语言模型智能分析，提供从数据加载、清洗、特征工程到EDA分析、统计检验、建模预测和报告生成的完整自动化工作。

二、核心功能模块
  功能模块	                  功能描述	                                       关键技术
|------------------------|-------------------------------------|-------------------------------------|
数据加载模块	支持CSV格式数据加载，自动识别数据结构和类型	          pandas, rich
数据清洗模块	自动化处理缺失值、重复值、异常值，数据标准化	       missingno, sklearn
特征工程模块	自动生成统计特征、分箱特征、标准化特征	              numpy, pandas
EDA分析模块	全面的探索性数据分析，包括单变量、双变量、多变量分析	   matplotlib, seaborn, plotly
统计检验模块	自动化执行正态性检验、相关性分析等统计检验	          scipy, statsmodels
机器学习建模	自动构建和评估机器学习模型，支持分类任务	          scikit-learn
智能LLM代理	基于DSPy的智能分析代理，提供数据洞察和建议	              DSPy, OpenAI API
报告生成模块	自动化生成结构化分析报告，支持Markdown格式	          markdown

三、系统架构
1. 整体架构
CWRU数据分析代理系统架构：
|------------------------|---------------------------|
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

四、技术栈详解
1. 核心框架
技术组件	 版本	       用途说明
|------|-------------|------------------------------------|
 Python	    3.11.0	     编程语言基础
 DSPy	    2.2.6	     声明式语言模型编程框架，构建智能代理
 OpenAI	    1.12.0	     GPT系列模型API接口
 Anthropic	0.25.1	     Claude模型API接口（可选）
2. 数据处理与分析
技术组件	  版本	             用途说明
|------|-------------|------------------------------------|
 pandas	      2.2.0	       核心数据结构与数据处理
 numpy	      1.26.4	   数值计算与数组操作
 scikit-learn 1.4.1.post1  机器学习算法库
 scipy	      1.12.0	   科学计算与统计工具
 statsmodels  0.14.1	   统计建模与分析
3. 数据可视化
技术组件	  版本	           用途说明
|----------|-------------|------------------|
 matplotlib	   3.8.3	    基础绘图库
 seaborn	   0.13.2	    统计图表美化
 plotly	       5.19.0	    交互式可视化图表
 missingno	   0.5.2	    缺失值可视化分析
4. 辅助工具
技术组件	版本	          用途说明
|------|-------------|------------------------|
 loguru	        0.7.2	     结构化日志记录
 rich	        13.7.0	     终端美化与进度显示
 python-dotenv	1.0.1	     环境变量管理
 tqdm	        4.66.2	     进度条显示

五、系统特色功能
1. 智能数据分析代理
DSPy签名系统：通过声明式签名定义分析任务，如DataAnalysisSignature、FeatureEngineeringSignature

链式思考（Chain of Thought）：通过dspy.ChainOfThought实现复杂推理过程

智能优化：使用BootstrapFewShot优化提示词和示例选择

2. 自动化工作流
完整分析流程：7步自动化流程，涵盖数据科学全生命周期

交互式模式：支持交互式探索，用户可逐步执行分析任务

批量处理模式：支持全自动批量分析，无需人工干预

3. 专业可视化系统
多图表类型：直方图、箱线图、相关性热图、PCA分析图等

交互式图表：基于Plotly的交互式HTML图表

专业报告：自动生成包含可视化引用的Markdown报告

4. 灵活配置系统
YAML配置文件：支持结构化配置管理

环境变量集成：通过.env文件管理敏感信息

多LLM支持：支持OpenAI GPT系列和Anthropic Claude模型

六、快速开始指南
1. 环境配置
bash
# 克隆项目
git clone <repository-url>
cd cwru-data-analysis-agent

# 创建虚拟环境（Windows）
python -m venv venv
venv\Scripts\activate

# 创建虚拟环境（Linux/Mac）
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
2. API密钥配置
创建.env文件：

env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
3. 配置文件设置
编辑config.yaml：

yaml
llm:
  provider: "openai"
  model: "gpt-4-turbo-preview"
  temperature: 0.1
  max_tokens: 2000
4. 运行分析
bash
# 完整分析模式
python run_analysis.py --data data_12k_10c.csv --mode full --output results

# 交互式分析模式
python run_analysis.py --data data_12k_10c.csv --mode interactive

七、项目目录结构
text
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

八、使用示例
1. 数据分析流程示例
python
# 初始化分析代理
agent = CWRUAnalysisAgent("config.yaml")

# 执行完整分析
report_path = agent.run_full_analysis(
    data_path="data_12k_10c.csv",
    output_dir="results"
)

print(f"分析报告已生成: {report_path}")
2. 交互式分析示例
python
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

九、扩展与定制
1. 添加新分析模块
python
# 1. 在src目录下创建新模块文件
# 2. 实现核心分析功能
# 3. 在主控制器中注册模块
# 4. 更新配置文件和运行脚本
2. 自定义LLM签名
python
class CustomAnalysisSignature(dspy.Signature):
    """自定义分析签名"""
    input_data = dspy.InputField(desc="输入数据描述")
    analysis_params = dspy.InputField(desc="分析参数", optional=True)
    
    analysis_results = dspy.OutputField(desc="分析结果")
    recommendations = dspy.OutputField(desc="建议")
    
# 在LLMAgent中注册使用
self.custom_analyzer = dspy.ChainOfThought(CustomAnalysisSignature)
3. 支持新数据格式
python
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

十、性能与优化
1. 性能特征
数据处理：支持百万级数据行的处理

内存管理：自动优化数据类型减少内存占用

并行处理：可扩展支持多进程/多线程处理

2. 优化建议
对于大型数据集，建议分批次处理

使用特征选择减少不必要特征

调整LLM模型的temperature参数控制输出稳定性

十一、故障排除
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

十二、贡献指南
欢迎贡献代码！请遵循以下步骤：

Fork本仓库

创建功能分支 (git checkout -b feature/AmazingFeature)

提交更改 (git commit -m 'Add some AmazingFeature')

推送到分支 (git push origin feature/AmazingFeature)

开启Pull Request

十三、许可证
本项目采用MIT许可证。详见LICENSE文件。

十四、致谢
Case Western Reserve University：提供轴承故障诊断数据集

OpenAI：提供GPT系列语言模型

DSPy团队：开发声明式语言模型编程框架

所有开源项目贡献者：提供优秀的基础库和工具

十五、联系方式
如有问题或建议，请通过以下方式联系：

项目地址：[GitHub Repository URL]

问题反馈：[Issue Tracker]

电子邮件：[renshuang@nuaa.edu.cn]

项目状态：活跃开发中
最后更新：2026年1月
版本：1.0.0

