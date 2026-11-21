# 增材制造领域问答系统（本地 RAG）

基于本地开源大模型的增材制造（Additive Manufacturing）领域问答系统：
在线按需获取公开资料 → 本地向量化入库 → 检索增强问答。

## 1. 功能

- 覆盖全领域工艺：FDM/FFF、SLA/DLP/LCD、SLS、SLM/LPBF、DED/WAAM、粘结剂喷射、材料、缺陷、后处理、设计与仿真
- **工艺标签可配置**：改 `amrag/processes.json` 即可增删工艺和检索关键词，无需改代码
- 中英文提问
- Ollama 本地运行 Qwen（生成）+ BGE-M3（向量）
- Hugging Face 数据集流式读取（`streaming=True`）+ OpenAlex 论文元数据与摘要
- ChromaDB 本地持久化索引，Streamlit 网页界面，答案附带来源

## 2. 目录结构

```
增材制造领域问答系统/
├─ app.py                  Streamlit 问答界面
├─ ingest.py               在线采集脚本
├─ requirements.txt
├─ .env.example            配置模板（复制为 .env）
├─ data/                   预留目录
└─ amrag/
   ├─ config.py            读取 .env 与工艺配置
   ├─ processes.json       工艺标签 + 关键词 + 检索式（可自由增删）
   ├─ ollama_client.py     Ollama /api/embed 与 /api/chat 封装
   ├─ online_sources.py    Hugging Face + OpenAlex 采集
   ├─ store.py             ChromaDB 增删查
   └─ rag.py               检索结果拼装 + 生成回答
```

## 3. 安装 Ollama（当前机器尚未安装）

1. 下载安装：https://ollama.com/download ，安装后确认终端里可以执行：

```powershell
ollama --version
```

2. 拉取模型（生成模型首次约 4-5 GB，向量模型约 1.2 GB）：

```powershell
ollama pull qwen2.5:7b
ollama pull bge-m3
ollama list
```

显存/内存较小可换成 `ollama pull qwen2.5:3b`，并把 `.env` 中 `LLM_MODEL` 改为 `qwen2.5:3b`。

Ollama 默认监听 `http://localhost:11434`，安装后需保持后台运行。

## 4. 创建环境

在项目根目录执行（本机默认 Python 3.12.4，也可用 `py -3.11` 新建虚拟环境）：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

若 PowerShell 禁止激活脚本：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

然后复制配置模板：

```powershell
Copy-Item .env.example .env
```

## 5. 导入在线资料

```powershell
python ingest.py --all                       # HF + OpenAlex，全部工艺
python ingest.py --openalex --process FDM --process SLA   # 只采集指定工艺
python ingest.py --list-processes            # 查看当前配置的工艺标签
```

参数：
- `--hf-limit`：每个 HF 配置最多读取的记录数（默认 300）
- `--openalex-per-query`：每个检索式最多获取的论文数（默认 30）
- `--process`：只采集指定工艺（可重复），不传则采集全部

首次向量化会调用本地 `bge-m3`，向量写入 `chroma_db/`。重复导入按内容 ID 更新，不会简单堆叠。

## 6. 启动问答界面

```powershell
streamlit run app.py
```

浏览器通常自动打开 `http://localhost:8501`。侧边栏「工艺范围」下拉框的内容直接来自 `processes.json`。

示例问题：
- FDM 打印过程中翘曲的主要原因是什么？
- SLA 后固化对树脂力学性能有什么影响？
- LPBF 中孔隙缺陷如何形成、如何抑制？
- How does powder recycling affect SLS part quality?
- Compare material extrusion and vat photopolymerization in process mechanism.

## 7. 增删工艺

编辑 `amrag/processes.json`，每个工艺包含：

```json
{
  "key": "WAAM",
  "name": "电弧增材制造",
  "priority": 90,
  "keywords": ["waam", "wire arc additive", "电弧增材"],
  "queries": ["wire arc additive manufacturing residual stress microstructure"]
}
```

- `key`：标签唯一标识，会写入向量库元数据，也是 `--process` 与界面下拉框的取值
- `name`：界面显示名称，也支持在 `--process` 中直接填写
- `priority`：关键词命中数相同时的判定优先级（数值大者优先）
- `keywords`：用于给采集内容打标签（小写子串匹配）
- `queries`：OpenAlex 检索式

改完后重新执行 `python ingest.py --all`，新工艺的资料就会入库；也可用 `PROCESS_FILE` 指向别处的配置文件。

## 8. 数据来源与局限

- Hugging Face Datasets streaming：https://huggingface.co/docs/datasets/stream
- 默认数据集：https://huggingface.co/datasets/ppak10/Additive-Manufacturing-Benchmark
- OpenAlex API：https://docs.openalex.org/
- Ollama：https://ollama.com/
- BGE-M3：https://huggingface.co/BAAI/bge-m3
- ChromaDB：https://docs.trychroma.com/

**关于“不下载原始数据”**：streaming 不会先下载完整数据集，但读取的数据仍会经网络传输；OpenAlex 只取元数据与摘要，不下载全文；Ollama 模型权重必须下载并存在本机；ChromaDB 在本机保存向量和文本片段。

**关于知识粒度**：摘要无法替代全文，细节参数类问题可能证据不足。系统被要求在这种情况下明确回答“证据不足”，而不是编造数值。要提升专业性，应补充有合法访问权限的全文、标准与设备手册，并保留 DOI/URL、年份与许可信息。

## 9. 常见问题

**Ollama 连接失败**：确认已安装并启动，终端执行 `ollama list`，浏览器访问 `http://localhost:11434/api/tags`。

**模型不存在**：`ollama pull qwen2.5:7b`、`ollama pull bge-m3`。

**Hugging Face 连接失败**：确认网络可访问 HF；也可先只跑 OpenAlex：`python ingest.py --openalex --openalex-per-query 5`。

**向量库为空**：先运行 `python ingest.py --all`，观察终端打印的采集与写入数量。

**选了某工艺但检索结果很少**：该工艺入库条目不足，先针对该工艺采集；通用（General）条目会在候选不足时自动补位。

## 10. 说明

这是一个可运行的 RAG 工程骨架，不是经过专家审核的问答语料库。若要用于研究或发表，仍需制定纳入标准、人工抽样审核、重复检测、答案溯源与许可审查。
