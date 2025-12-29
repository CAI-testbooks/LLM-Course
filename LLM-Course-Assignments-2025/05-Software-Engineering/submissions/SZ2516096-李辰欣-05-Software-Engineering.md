 # **AutoDev-Agent——基于大模型的自动化软件开发代理系统**  

---

## 一、实验题目  
基于大模型的自动化软件开发代理系统系统设计与原型实现  

> 代码仓库链接：[LLM-Course课设仓库：AutoDev-Agent](https://github.com/lcx4411/AutoDev-Agent)

---

## 二、实验目的  

1. 理解并掌握大模型在软件开发自动化中的应用场景与实现方法。  
2. 设计并实现一个基于Agent架构的AI辅助编程系统，支持需求理解、代码生成、测试生成、错误修复等功能。  
3. 通过实际数据集（HumanEval、MBPP、SWE-Bench Lite）验证系统各模块的有效性。  
4. 探索并实践Self-Reflection机制在AI辅助开发中的迭代优化作用。  

---

## 三、实验内容与系统架构  

### 3.1 系统目标  
构建一个**多Agent协同**的AI开发助手，具备以下能力：  
- 自然语言需求理解与任务拆解  
- 函数/文件级代码生成  
- 单元测试自动生成  
- 基于错误信息的Bug定位与修复  
- 自我反思与迭代优化  

### 3.2 系统架构  
系统采用分层Agent协作架构，具体流程如下：  

```
        ┌──────────────┐
        │   用户需求   │
        └──────┬───────┘
               ↓
        ┌──────────────┐
        │ Task Planner │ ← 需求理解 / 任务拆解
        └──────┬───────┘
               ↓
     ┌─────────┴─────────┐
     ↓                   ↓
Code Generator     Test Generator
     ↓                   ↓
┌──────────────┐   ┌──────────────┐
│ 代码执行环境  │←→ │ 单元测试运行 │
└──────┬───────┘   └──────┬───────┘
       ↓                  ↓
   Error Trace      Test Failure
        └──────┬───────┘
               ↓
        ┌──────────────┐
        │ Bug Fix Agent│ ← 反思 + 修复
        └──────────────┘
```

### 3.2 项目结构

```text
code-assistant-agent/
├── README.md                    # 项目说明文档
├── .gitignore                   # Git忽略文件
│
├── config/                     # 配置文件目录
│   ├── __init__.py
│   ├── config.yaml            # 主配置文件
│   ├── model_config.yaml      # 模型配置
│   └── prompt_templates.yaml  # 提示词模板
│
├── src/                        # 源代码目录
│   ├── __init__.py
│   ├── main.py                # 主入口文件
│   │
│   ├── core/                  # 核心逻辑
│   │   ├── __init__.py
│   │   ├── agent.py          # 主Agent类
│   │   ├── reflection.py     # 反思引擎
│   │   ├── planner.py        # 任务规划器
│   │   ├── code_generator.py # 代码生成器
│   │   ├── tester.py         # 测试生成器
│   │   └── fixer.py          # Bug修复器
│   │
│   ├── models/                # 模型相关
│   │   ├── __init__.py
│   │   ├── base_model.py     # 模型基类
│   │   └── qwen_model.py     # Qwen适配器
│   │
│   ├── tools/                 # 工具模块
│   │   ├── __init__.py
│   │   ├── base_tool.py      # 工具基类
│   │   └── python_repl.py    # Python执行器
│   │
│   ├── utils/                 # 工具函数
│   │   ├── __init__.py
│   │   ├── llm_utils.py      # LLM调用
│   │   └── config_utils.py   # 配置加载
│   
```

---

## 四、实验步骤与实现方案  

### 4.1 数据准备  
使用开源代码数据集进行模块验证：  
- **HumanEval**：用于代码生成任务评估  
- **MBPP**：用于测试生成任务评估  
- **SWE-Bench Lite**：用于Bug修复任务评估  

> 数据集通过HuggingFace直接加载，便于复现与实验。

### 4.2 模型选择  
- **基础模型**：Qwen3-Coder-7B/14B（代码生成与修复）  
- **备用模型**：DeepSeek-Coder-6.7B（轻量快速场景）  
- **推理方式**：Prompt Engineering + Tool-Calling  

### 4.3 核心Agent设计  

#### 4.3.1 Task Planner  
将自然语言需求拆解为结构化开发任务，输出JSON格式任务描述。  

#### 4.3.2 Code Generator Agent  
基于任务描述生成符合规范的代码，支持多语言（优先Python）。  

#### 4.3.3 Test Generator Agent  
针对生成的代码自动生成单元测试用例，覆盖正常、边界与异常情况。  

#### 4.3.4 Bug Fix Agent  
结合错误信息与测试失败信息，分析原因并生成修复后的代码。  

### 4.4 Reflection Agent
引入Self-Reflection Prompt，使系统能在失败后分析原因、调整假设，并进行多轮迭代优化。  

---

## 五、实验进展与当前版本说明  

### 5.0 历史版本（第一版）  
- 已完成系统整体架构设计与各Agent功能定义  
- 完成Prompt模板设计与数据集选型  
- 实现模拟数据流，验证系统逻辑可行性

### 5.1 当前版本（第二版）
- 参考标准项目开发流程，构建规范的项目结构，后续将基于该版本进行功能完善。已将第一版内容完成迁移。

### 5.2 已实现功能  
- Task Planner 结构化输出  
- Code / Test / Bug Fix Agent 的Prompt模板  
- Self-Reflection 机制设计  

### 5.3 待实现与优化  
1. 完善各Agent的具体实现与模型调用  
2. 实现评估指标计算（Pass@1、测试覆盖率、修复成功率等）  
3. 模块解耦，支持各Agent独立运行与测试  
4. 开发CLI工具，提供用户交互界面  

---

## 六、实验评估方案  

| 任务              | 评估指标     | 说明                     |
|------------------|-------------|--------------------------|
| 代码生成          | Pass@1      | 首次生成即通过测试的比例     |
| 测试生成          | 代码覆盖率    | 生成测试对代码的覆盖情况     |
| Bug修复           | 修复成功率    | 成功修复的Bug比例          |
| 系统整体          | 平均迭代次数  | 从生成到最终通过所需迭代次数   |

--- 

> 说明：本报告为第二版实验报告，后续将随系统迭代持续更新完善。
