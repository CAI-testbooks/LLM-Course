# Juris-RAG：法律领域智能问答系统

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)](https://langchain.com)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 项目简介

**Juris-RAG** 是一个基于检索增强生成（RAG）技术的中文法律领域智能问答系统。该系统整合了刑法法条和司法案例数据，通过向量检索和大语言模型，为用户提供准确、专业、可追溯的法律咨询服务。

### 核心特性

| 特性 | 描述 |
|------|------|
| 领域知识库 | 整合刑法法条 + CAIL 刑事案例 |
| 语义检索 | 基于 BGE-M3 向量模型的语义匹配 |
| 智能生成 | Qwen2.5-7B-Instruct 生成专业回答 |
| 多轮对话 | 支持上下文理解与连续追问 |
| 引用与置信度 | 回答展示引用来源与置信度侧栏 |
| 文档搜索模式 | 不经 LLM 的直检索模式 |
| Web 交互 | ChatGPT 风格布局，Enter 发送，Shift+Enter 换行 |
| 速率限制与重试 | 可配置 RPM/TPM 限制，批处理节流 |
| 评估脚本 | Accuracy、Citation F1、Hallucination 等指标 |

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      用户界面 (Gradio)                       │
├─────────────────────────────────────────────────────────────┤
│                     RAG引擎 (rag_engine.py)                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ 问题改写    │→│ 向量检索     │→│ 答案生成 + 引用      │ │
│  │ (多轮对话)  │  │ (Top-K)     │  │ (Qwen2.5)          │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                  向量数据库 (ChromaDB)                       │
│  ┌─────────────────────┐  ┌───────────────────────────────┐│
│  │ 刑法法条 (Statute)  │  │ CAIL案例 (Case) - 20k+条      ││
│  └─────────────────────┘  └───────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## 项目结构

```
Juris-RAG/
├── app.py                 # Gradio Web应用入口
├── eval.py                # 评估脚本
├── requirements.txt       # 依赖列表
├── README.md              # 项目说明
├── .env                   # 环境变量配置（需自建）
├── .gitignore             # Git忽略配置
│
├── assets/                # 前端样式
│   └── ui.css             # 自定义界面样式
│
├── src/                   # 源代码目录
│   ├── __init__.py
│   ├── config.py          # 配置参数
│   ├── data_processing.py # 数据处理与向量化
│   ├── cail_adapter.py    # CAIL数据文件选择适配
│   └── rag_engine.py      # RAG核心引擎
│
├── data/                  # 数据目录
│   ├── raw/               # 原始数据
│   │   ├── criminal_code.txt
│   │   ├── cail_cases_20k.json
│   │   └── cail_cases.json
│   ├── eval/              # 评估数据
│   └── vector_db/         # 向量数据库（自动生成）
│
└── reports/               # 实验报告目录
    └── 学号-姓名-01-NLP.md
```

## 快速开始

### 1. 环境准备

```bash
git clone https://github.com/your-username/Juris-RAG.git
cd Juris-RAG

python -m venv venv

# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. 配置 API Key

创建 `.env` 文件：

```env
SILICONFLOW_API_KEY=your_api_key_here
```

或直接设置环境变量：

```bash
# Windows PowerShell
$env:SILICONFLOW_API_KEY="your_api_key_here"

# Windows CMD
set SILICONFLOW_API_KEY=your_api_key_here

# Linux/Mac
export SILICONFLOW_API_KEY=your_api_key_here
```

### 3. 准备数据

确保 `data/raw/` 目录下有以下文件：
- `criminal_code.txt` - 刑法文本
- `cail_cases_20k.json` - CAIL 案例精简数据（JSON Lines 格式，推荐）
- `cail_cases.json` - 原始案例数据（可选）

系统会自动优先使用 `cail_cases_20k.json`，若不存在则回退为 `cail_cases.json`。

### 4. 构建向量数据库

```bash
python -m src.data_processing
```

输出示例：
```
正在加载法条: ./data/raw/criminal_code.txt
加载法条完成，共 XXX 个文档块
正在加载 CAIL 案例: ./data/raw/cail_cases_20k.json (限制 20000 条)
加载案例完成，共 20000 个文档

数据集统计:
  总文档数: 5XXX
  按类型分布: {'statute': XXX, 'case': 20000}
  平均长度: XXX.X 字符

准备向量化 20XXX 条文档...
向量数据库构建完成！已保存至 ./data/vector_db
```

### 5. 启动 Web 应用

```bash
python app.py
```

访问 `http://127.0.0.1:7860` 开始使用。

### 6. 运行评估

```bash
python eval.py
```

## 配置说明

主要配置项在 `src/config.py`：

```python
# 模型配置
EMBEDDING_MODEL = "BAAI/bge-m3"
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# RAG参数
CHUNK_SIZE = 500
RETRIEVAL_TOP_K = 5
LLM_TEMPERATURE = 0.1

# 长上下文支持
MAX_CONTEXT_LENGTH = 32768
MAX_HISTORY_TURNS = 10

# 向量化速率限制
EMBED_RPM_LIMIT = 2000
EMBED_TPM_LIMIT = 500000
EMBED_BATCH_SIZE = 20
EMBED_SLEEP_SECONDS = 0.1
```

## 使用示例

### 基础问答

```
用户: 故意杀人罪怎么判刑？

助手: 根据《刑法》第二百三十二条规定，故意杀人的，处死刑、无期徒刑或者
十年以上有期徒刑；情节较轻的，处三年以上十年以下有期徒刑。

引用来源:
[1] 中华人民共和国刑法 (statute)
    条款: 第二百三十二条
```

### 多轮对话

```
用户: 盗窃罪怎么判？
助手: 根据《刑法》第二百六十四条...

用户: 如果数额特别巨大呢？
助手: 数额特别巨大或者有其他特别严重情节的...
```

### 拒绝不确定问题

```
用户: 民法典关于合同的规定是什么？

助手: 抱歉，根据现有法律数据库，我无法准确回答此问题。
本系统主要涵盖刑法相关内容，民法典相关问题建议咨询专业律师。
```

## 评估指标

| 指标 | 说明 | 基线值 |
|------|------|--------|
| Accuracy | 回答正确率 | ~70% |
| Citation F1 | 引用来源准确性 | ~80% |
| Hallucination Rate | 幻觉率（越低越好）| <10% |
| Avg Latency | 平均响应时间 | <3s |

## 优化方向

- 扩展数据源（民法典、行政法等）
- 引入重排序模型提升检索质量
- LoRA 微调提升领域准确率
- 添加更多评估基准（LegalBench 等）
- 支持文档上传与实时索引

## 技术栈

| 组件 | 技术选型 |
|------|----------|
| 框架 | LangChain 0.1+ |
| 向量库 | ChromaDB |
| Embedding | BAAI/bge-m3 (via SiliconFlow) |
| LLM | Qwen/Qwen2.5-7B-Instruct |
| 前端 | Gradio 4.0+（兼容 6.x） |
| API | SiliconFlow |

## 数据来源

1. **刑法法条**：中华人民共和国刑法（2020 修正）
2. **司法案例**：CAIL2018 中国法律智能挑战赛数据集

## 免责声明

本系统仅供学习和研究使用，不构成法律建议。如有实际法律问题，请咨询专业律师。

## 许可证

MIT License

## 作者

**[你的姓名]** - [学号]
