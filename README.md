# 基于 RAG 的医疗问答系统

基于检索增强生成（RAG）技术的中文医疗问答系统，支持查询重写、HyDE、重排序等检索优化技术。

## 目录

- [代码结构](#代码结构)
- [运行环境](#运行环境)
- [依赖安装](#依赖安装)
- [快速开始](#快速开始)
- [完整流程复现](#完整流程复现)
  - [1. 数据处理](#1-数据处理)
  - [2. 推理服务](#2-推理服务)
  - [3. 评估脚本](#3-评估脚本)
  - [4. 前端部署](#4-前端部署)
- [API 文档](#api-文档)
- [配置说明](#配置说明)

---

## 代码结构

```
.
├── backend/                    # 后端代码
│   ├── rag.py                 # RAG 核心模块（Embedding、检索、重写、重排序）
│   ├── main.py                # FastAPI 服务入口
│   ├── prompts.py             # 提示词模板
│   ├── evaluate.py            # 评估脚本
│   ├── ingest.py              # 数据导入脚本
│   └── __init__.py            # 模块导出
├── frontend/                   # Vue 前端代码
│   ├── src/
│   │   ├── App.vue            # 根组件
│   │   ├── main.js            # Vue 入口
│   │   ├── api/index.js       # API 封装
│   │   ├── stores/chat.js     # Pinia 状态管理
│   │   ├── views/ChatView.vue # 聊天视图
│   │   ├── components/        # 组件目录
│   │   └── assets/main.css    # 全局样式
│   ├── package.json
│   └── vite.config.js
├── data/
│   ├── faiss_db/              # FAISS 向量索引存储
│   └── sample/                # 示例数据
└── README.md
```

### 核心模块说明

| 文件 | 功能 |
|------|------|
| `rag.py` | RAG 核心实现：Embedding、Chunker、VectorStore、QueryRewriter、Reranker、LLM |
| `main.py` | FastAPI 后端服务，提供 REST API |
| `prompts.py` | 系统提示词和查询重写模板 |
| `evaluate.py` | 检索和生成质量评估，支持多配置对比 |
| `ingest.py` | 知识库数据导入工具 |

---

## 运行环境

### 系统要求

- **操作系统**: Linux / macOS / Windows
- **Python**: 3.10+
- **Node.js**: 18+ (前端)
- **内存**: 8GB+ (推荐 16GB)
- **GPU**: 可选，用于加速 Embedding 和 Reranker

### 测试环境

```
- Ubuntu 22.04 LTS
- Python 3.12
- CUDA 12.1 (可选)
```

---

## 依赖安装

### 方式一：使用 uv（推荐）

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

cd backend
uv sync
```

### 方式二：使用 pip

```bash
cd backend
conda create -n rag python==3.12

pip install -r requirements.txt
```

### 前端依赖

```bash
cd frontend
npm install
# 或
pnpm install
```

---

## 快速开始

### 1. 导入示例数据

```bash
cd backend
python ingest.py --create-sample
```

### 2. 启动后端服务

```bash
python main.py \
    --provider deepseek \
    --api-key YOUR_API_KEY \
    --model deepseek-chat
```

### 3. 启动前端（可选）

```bash
cd frontend
npm run dev
```

### 4. 访问服务

- **API 文档**: http://localhost:8080/docs
- **前端界面**: http://localhost:3000

---

## 完整流程复现

### 1. 数据处理

#### 1.1 准备原始数据

支持以下数据格式：

**JSON 格式（问答对）**:
```json
[
  {"question": "什么是高血压？", "answer": "高血压是一种常见的慢性疾病..."},
  {"question": "糖尿病症状有哪些？", "answer": "典型症状包括多饮、多食、多尿..."}
]
```

**JSON 格式（文档）**:
```json
[
  {"title": "高血压指南", "content": "高血压的诊断标准..."},
  {"content": "糖尿病是一种代谢疾病..."}
]
```

**纯文本文件**: `.txt`, `.md`

#### 1.2 导入数据到知识库

```bash
# 导入 JSON 文件
python ingest.py --json data/medical_qa.json --db-dir data/faiss_db

# 导入单个文件
python ingest.py --file data/document.txt --db-dir data/faiss_db

# 导入整个目录
python ingest.py --dir data/documents --db-dir data/faiss_db

# 指定 Embedding 模型
python ingest.py --json data/qa.json \
    --embedding-model BAAI/bge-base-zh-v1.5 \
    --db-dir data/faiss_db
```

#### 1.3 数据处理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--db-dir` | `data/faiss_db` | 向量索引存储目录 |
| `--embedding-model` | `BAAI/bge-base-zh-v1.5` | Embedding 模型 |
| `--extensions` | `.txt .md .pdf` | 扫描的文件扩展名 |

---

### 2. 推理服务

#### 2.1 启动 API 服务

```bash
python main.py \
    --provider deepseek \
    --api-key YOUR_API_KEY \
    --model deepseek-chat \
    --db-dir data/faiss_db \
    --embedding BAAI/bge-base-zh-v1.5 \
    --top-k 5 \
    --enable-rewrite \
    --rewrite-mode single \
    --enable-cache \
    --cache-size 1000 \
    --enable-rerank \
    --rerank-model BAAI/bge-reranker-base \
    --host 0.0.0.0 \
    --port 8080
```

#### 2.2 命令行参数说明

| 参数 | 说明 | 可选值 |
|------|------|--------|
| `--provider` | LLM 提供商 | `openai`, `deepseek`, `zhipu`, `moonshot`, `qwen`, `ollama` |
| `--api-key` | API 密钥 | - |
| `--model` | 模型名称 | 如 `deepseek-chat`, `gpt-4`, `glm-4` |
| `--base-url` | 自定义 API 地址 | - |
| `--db-dir` | 向量索引目录 | - |
| `--embedding` | Embedding 模型 | HuggingFace 模型名 |
| `--top-k` | 检索数量 | 1-20 |
| `--enable-rewrite` | 启用查询重写 | - |
| `--rewrite-mode` | 重写模式 | `single`, `multi`, `context`, `auto`, `hyde` |
| `--enable-rerank` | 启用重排序 | - |
| `--rerank-model` | 重排序模型 | HuggingFace 模型名 |
| `--enable-cache` | 启用缓存 | - |
| `--cache-size` | 缓存大小 | 默认 1000 |

### 3. 评估脚本

#### 3.1 准备评估数据

评估数据格式（JSON）:
```json
[
  {
    "question": "高血压有什么症状？",
    "answer": "高血压早期通常没有明显症状...",
    "relevant_docs": ["doc_001", "doc_002"]
  }
]
```

#### 3.2 运行单配置评估

```bash
python evaluate.py \
    --provider deepseek \
    --api-key YOUR_API_KEY \
    --model deepseek-chat \
    --db-dir data/faiss_db \
    --eval-data data/eval_set.json \
    --output results/eval_result.json \
    --top-k 5 \
    --enable-rewrite \
    --rewrite-mode single \
    --enable-rerank \
    --max-samples 100
```

#### 3.3 运行多配置对比评估

```bash
python evaluate.py \
    --provider deepseek \
    --api-key YOUR_API_KEY \
    --db-dir data/faiss_db \
    --eval-data data/eval_set.json \
    --compare \
    --max-samples 100
```

对比模式会自动测试以下 7 种配置：

| 配置 | 说明 |
|------|------|
| baseline | 无重写、无重排序 |
| rewrite_single | 单查询重写 |
| rewrite_multi | 多查询扩展 |
| rewrite_hyde | HyDE 假设文档 |
| rerank_only | 仅重排序 |
| rewrite_single_rerank | 单查询重写 + 重排序 |
| rewrite_hyde_rerank | HyDE + 重排序 |

#### 3.4 评估指标

| 指标 | 说明 |
|------|------|
| Hit@K | Top-K 结果中包含相关文档的比例 |
| MRR | 平均倒数排名 |
| NDCG@K | 归一化折损累积增益 |
| Latency | 平均响应时间 |

#### 3.5 评估参数说明

| 参数 | 说明 |
|------|------|
| `--eval-data` | 评估数据集路径 |
| `--output` | 结果输出路径 |
| `--max-samples` | 最大评估样本数 |
| `--compare` | 启用多配置对比模式 |
| `--enable-judge` | 启用 LLM 评判（较慢） |

---

### 4. 前端部署

#### 4.1 开发模式

```bash
cd frontend
npm install
npm run dev
```

访问 http://localhost:3000


