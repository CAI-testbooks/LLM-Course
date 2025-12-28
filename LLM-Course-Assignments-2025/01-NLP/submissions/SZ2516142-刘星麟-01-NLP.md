# [阶段性报告] 基于 RAG 的法律领域智能问答系统 (Juris-RAG)

## 0. 项目进展简述 (Phase 1→2 中途状态)

- **当前状态**：✅ Web UI、评估与端到端链路已跑通。完成 Gradio 前端（聊天/搜索/信息页）、评估器（`eval.py`）、向量库构建（`data/vector_db/`）、RAG 引擎完善（引用与置信度），并通过 CLI + Web 双通路测试。
- **本次推进亮点**：
   - Gradio 多标签界面：智能问答、文档搜索、系统信息；支持置信度与引用展示。
   - 评估管线：内置评测集与指标（Accuracy/Citation F1/Hallucination/Latency），自动保存 JSON 报告至 `reports/`。
   - 向量化稳健性：批处理 + 指数回退，规避 API 限速；已生成 `chroma.sqlite3` 与持久化目录。
   - 拒答机制：低置信度与越界问题明确拒答，降低幻觉风险。
- **已完成工作**：
   - 数据清洗（ETL）：刑法法条与 CAIL 案例的清洗、分块与向量化（BGE-M3）。
   - 向量数据库：基于 ChromaDB 的法条+案例双库，持久化于 `data/vector_db/`。
   - 核心引擎：LangChain 历史感知检索链 + 生成链，引用来源与置信度计算。
   - 前端与交互：Gradio Web 应用；引用展示与置信度提示；文档搜索 Tab。
- **仓库链接**：[GitHub - Juris-RAG](https://github.com/liuxinglin70-cmyk/Juris-RAG)
- **本次 PR 内容**：补充 Web UI、评估脚本、配置与错误处理逻辑。

------

## 一、项目概述

### 1.1 项目背景

在法律咨询场景中，普通用户难以理解晦涩的法条，而通用大模型（LLM）常出现编造法条（幻觉）的问题。本项目旨在构建 **Juris-RAG**，通过检索增强生成技术，结合《中华人民共和国刑法》与真实司法案例，提供准确、可解释的法律问答服务。

### 1.2 项目目标

- **核心功能**：构建支持长上下文、多轮对话的 RAG 系统。
- **领域对齐**：基于刑法和 CAIL 数据集，解决通用模型在法律领域的幻觉问题。
- **可解释性**：回答必须精准引用法条来源（精确到条款号）。

### 1.3 技术选型 (已落地)

| **组件**          | **选型**            | **理由**                                                     |
| ----------------- | ------------------- | ------------------------------------------------------------ |
| **LLM**           | Qwen2.5-7B-Instruct | (Via SiliconFlow API) 开源界中文逻辑最强模型之一，支持长文本。 |
| **Embedding**     | BAAI/bge-m3         | 智源出品，目前中文语义检索的 SOTA 模型。                     |
| **Vector DB**     | ChromaDB            | 轻量级本地向量库，无需复杂部署，适合快速迭代。               |
| **Orchestration** | LangChain           | 利用其 `create_history_aware_retriever` 实现多轮对话管理。   |

------

## 二、数据来源与处理

### 2.1 数据来源

本项目采用 **“法条+案例”双库驱动** 策略，数据清单如下：

1. **核心法条库 (Statutes)**：
   - 来源：**《中华人民共和国刑法》（2020修正版）**。
   - 处理：格式转换为 UTF-8 纯文本，确保编码兼容性。
2. **参考案例库 (Cases)**：
   - 来源：**CAIL 2018 (China AI Law Challenge)** 公开数据集。
   - 选取：使用 `first_stage/train.json` 中的刑事案件数据。
   - 规模：目前测试阶段选取 2000-5000 条高质量数据进行向量化。

### 2.2 数据处理流程 (ETL)

已实现自动化脚本 `src/data_processing.py`，流程如下：

1. **清洗与加载 (Loading)**：
   - 法条：读取 `.txt`，按“条”进行物理分割。
   - 案例：读取 `.json`，提取 `fact` (案情) 作为检索文本，提取 `meta` (判决结果) 作为元数据。
2. **分块 (Chunking)**：
   - 使用 `RecursiveCharacterTextSplitter`。
   - **策略**：`chunk_size=500`, `chunk_overlap=100`。
   - *设计考量*：保证每一条法律条文的语义完整性，避免断章取义。
3. **向量化 (Embedding)**：
   - 调用 `bge-m3` 模型将文本转化为 1024 维向量，并持久化存储至 `data/vector_db`。

------

## 三、方法与实现

### 3.1 系统架构设计

系统分为三层架构，目前已完成后端引擎层的开发：

```
[用户输入] 
    ↓
[历史对话重写器 (Contextualize Chain)] <--- 创新点：多轮对话优化
    ↓ (独立化问题)
[混合检索器 (Hybrid Retriever)]
    ↓ 1. 检索法条库 (Statutes)
    ↓ 2. 检索案例库 (Cases)
[重排序 (Rerank)] (计划中)
    ↓ (Top-K Docs)
[生成器 (Generator)] -> Prompt: "基于上下文回答并引用来源[1]"
    ↓
[最终回答]
```

### 3.2 关键代码实现逻辑

1. **多轮对话记忆**：
   - 利用 LangChain 的 `create_history_aware_retriever`，将用户后续的追问（如“那判几年？”）结合历史记录重写为完整问题（如“故意杀人罪判几年？”），再进行检索。
2. **引用来源追踪**：
   - 在 Metadata 中保留了 `source` (来源文件名) 和 `id` (条款号)。
   - Prompt 中强制要求模型输出 `[Source ID]`，并在后端解析显示。

------

## 四、初步实验结果 (Preliminary Results)

*注：当前处于基础建设阶段，以下为代码跑通测试结果，大规模评估表格将在下一阶段补充。*

### 4.1 命令行测试 (CLI Test)

目前通过 `src/rag_engine.py` 进行的单元测试效果如下：

> User: 故意杀人怎么判刑？
>
> AI: 根据《中华人民共和国刑法》第二百三十二条，故意杀人的，处死刑、无期徒刑或者十年以上有期徒刑；情节较轻的，处三年以上十年以下有期徒刑。[来源: 刑法]
>
> User: 那如果是情节较轻的呢？ (测试多轮对话)
>
> AI: 对于故意杀人情节较轻的情况，法律规定处三年以上十年以下有期徒刑。[来源: 刑法]

### 4.2 评估计划 (Evaluation Plan)

下一阶段将运行 `eval.py`，预计收集以下指标（表格模板）：

| **评估维度** | **指标**          | **目标值** | **当前状态** |
| ------------ | ----------------- | ---------- | ------------ |
| **检索质量** | Recall@5          | > 85%      | 待评测       |
| **生成质量** | 准确率 (Accuracy) | > 80%      | 待评测       |
| **引用规范** | 引用 F1 Score     | > 75%      | 待评测       |
| **系统性能** | 平均响应时间      | < 3s       | ~2.5s (API)  |

### 4.3 Web UI 测试 (Gradio)

当前 Web 端已支持：

- 置信度提示：以颜色与百分比显示（阈值控制不确定回答）。
- 引用展示：按来源与元数据（罪名/条款）列出，便于追溯。
- 文档搜索：不经 LLM，直接返回 Top-K 相似文档。

示例（界面交互与 CLI 一致）：

> 问题：盗窃罪的量刑标准是什么？
>
> 回答：根据《刑法》相关规定……并附 [来源] 与置信度。

------

## 五、问题分析与创新点

### 5.1 遇到的挑战与解决方案

1. **编码问题**：
   - *问题*：下载的刑法文档为 docx 且默认 GBK 编码，导致向量化时出现乱码。
   - *解决*：编写了转换脚本，强制转换为 UTF-8 格式，并清洗了无关的格式符号。
2. **API 速率限制**：
   - *问题*：批量写入 5000 条数据时触发 API Rate Limit。
   - *解决*：在 `data_processing.py` 中增加了 batch 处理和 `time.sleep` 缓冲机制。

### 5.2 架构创新点

1. **双库检索机制**：区别于传统的单库 RAG，本项目将“法条（硬规则）”和“案例（软参考）”在逻辑上区分，Prompt 优先采用法条，案例作为解释补充。
2. **流式输出集成**：后端引擎原生支持 Streaming，为前端打字机效果做好了准备。

------

## 六、Demo 演示

### 6.1 代码仓库

项目已托管至 GitHub，支持一键复现：

https://github.com/liuxinglin70-cmyk/Juris-RAG

### 6.2 运行方式

Bash

```
# 1. 安装依赖
pip install -r requirements.txt
# 2. 设置 API Key
export SILICONFLOW_API_KEY=sk-xxxx
# 3. 运行数据处理
python src/data_processing.py
# 4. 启动 Web UI（Gradio）
python app.py
# 5. 启动命令行测试（可选）
python src/rag_engine.py
```

*(已提供 Gradio Web UI：访问 http://localhost:7860)*

PowerShell（Windows）

```
# 1. 安装依赖
pip install -r requirements.txt
# 2. 设置 API Key（当前会话）
$env:SILICONFLOW_API_KEY="sk-xxxx"
# 3. 构建向量库
python -m src.data_processing
# 4. 启动 Web UI
python app.py
```

------

## 七、未来改进方向 (Next Steps)

围绕性能与可靠性继续迭代：

1. **检索质量优化**：集成 Rerank 模型（如 Cohere/RAG-Rerank 或 Cross-Encoder），提升 Top-K 命中与 Citation F1。
2. **数据规模扩展**：将案例库扩展至 10k+，观测 Recall/Latency 变化并更新评估报告。
3. **评估深化**：完善 `eval.py` 的类别细分与曲线绘制，自动导出 `reports/eval_report_*.json` 并附可视化。
4. **鲁棒性与容错**：细化速率限制与重试策略，补充日志追踪与异常上报。
5. **可能的微调**：尝试 LoRA/提示优化，降低幻觉率，提高刑法场景准确率。
