
# 06-Data-Processing：AI驱动的大规模数据分析代理
本项目是「方向06：数据处理及统计综合作业」的实现，基于LLM Agent构建了全自动化数据分析工作流，支持从原始数据到洞见报告的端到端流程。

## 一、项目概述
实现了**AI驱动的数据分析代理**，核心能力是接收原始数据集后，自动完成数据清洗、EDA分析、特征工程、统计建模、可视化及洞见报告生成，最终输出PDF格式的分析报告。

### 📂项目结构

```
.
├── agents/                       #多 Agent 分工模块（数据理解、代码生成、报告生成等）
│   ├── __pycache__/
│   ├── code_execution_agent.py   
│   ├── content_planning_agent.py
│   ├── plan_analytics.py         
│   ├── report_generation.py     
│   └── understand_data.py        
├── data/                         #原始数据集 + 数据 schema（数据结构描述）
│   ├── df/                      
│   └── schema/                  
├── output/                       # 生成的分析报告（PDF）
├── plots/                        # 分析生成的可视化图表
├── prompts/
│   ├── code_correction_prompt_template.txt
│   ├── code_generation_prompt_template.txt
│   ├── data_understanding_prompt.txt
│   ├── interpretation_prompt.txt
│   └── plotting_prompt_template.txt
├── utils/                        #工具函数集合（加载数据、初始化 LLM、数据预处理）
│   ├── load_data.py            
│   ├── load_llm.py               
│   └── preprocess_data.py       
├── myenv1/                        # 虚拟环境                 
├── .gitignore
├── main.py                       #项目入口，启动整个 AI 数据分析工作流
├── README.md
├── requirements.txt              #项目依赖包清单
├── state.py                      #定义 LangGraph 的共享状态，传递 Agent 间数据
└── workflow.py                   # 构建 LangGraph 多 Agent 工作流的核心配置
```

## 二、作业要求匹配模块
### 1. 数据准备
- 数据集选择：采用**公开大规模医疗数据集**（data/NSCLC_data.csv`，非小细胞肺癌临床数据集，符合公开数据集要求）；
- 数据管理：通过 `utils/load_data.py` 实现数据集自动加载，`utils/preprocess_data.py` 支持数据格式兼容处理。


### 2. 模型选型
- 核心框架：基于 **LLM Agent + LangGraph** 构建多智能体工作流（对应作业要求的LangGraph/DSy框架）；
- 工具集成：
  - 基础工具：Pandas（数据处理）、Statsmodels（统计建模）、Matplotlib/Seaborn（可视化）；
  - Agent组件：通过 `agents/` 目录下的多智能体分工实现功能（数据理解Agent、分析规划Agent、代码执行Agent、报告生成Agent等）。


### 3. 核心功能
100%覆盖作业要求的核心功能：
- 自动清洗：通过 `agents/understand_data.py` 识别数据缺失、异常值，自动执行去重、填充等操作；
- 特征工程：Agent自动分析变量类型，生成衍生特征（如临床指标的分组统计特征）；
- 可视化：通过 `agents/plan_analytics.py` 自动生成多维度图表（如生存分析图、变量分布直方图，输出至 `plots/` 目录）；
- 统计检验：集成Statsmodels实现卡方检验、生存分析等统计方法；
- 洞见总结：通过 `agents/report_generation.py` 提炼数据结论，转化为自然语言洞见。


### 4. 迭代优化
- Prompt升级：在 `prompts/` 目录下维护了精细化的Prompt模板（如代码生成、纠错、洞见解释模板）；
- 工具增强：扩展了Pandas的高级数据处理方法、Statsmodels的统计模型库，提升分析深度。


### 5. 部署
支持**上传CSV自动输出报告**：
1. 将待分析的CSV文件放入 `data/df/` 目录；
2. 运行 `main.py`，工作流自动执行全流程；
3. 最终报告输出至 `output/` 目录（包含Markdown和PDF格式）。


### 6. 评估
采用作业要求的**人工评估**：
- 报告准确性：验证分析结论与数据实际规律的一致性；
- 洞见深度：评估报告中对临床指标关联、生存因素的解读维度。


## 三、快速开始
1. 克隆本仓库：
   ```bash
   git clone https://github.com/Anita0116/-06-Data-.git
   cd -06-Data-/agentic-ai-data-analyst

