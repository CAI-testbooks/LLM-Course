# 中医中药 Hybrid GraphRAG 问答系统 — 迭代 1（数据工程与 Baseline RAG）

- **学号**：`SX2624013`
- **姓名**：`刘瀚远`
- **方向**：`01-NLP`（构建领域特定 RAG 问答系统）
- **独立代码仓库**：[https://github.com/lhy-0317/TCM-KGQA](https://github.com/lhy-0317/TCM-KGQA)
- **当前版本**：`v0.1.0`

---

## 一、阶段目标与演进基础

本项目以此前《知识表示》课程的单跳问答系统原型为基础进行工程化重构与领域升级：将原有面向动漫角色的 `BERT-NER + py2neo + MiniLM + Qwen3-1.7B` 原型迁移至**中医中药（TCM）**垂直领域，并全面升级模型配置与检索评估基础设施。

本阶段重点完成：
1. **工程架构重构与硬编码清理**：剔除原代码中所有本机绝对路径硬编码与废弃依赖，建立 `configs/config.yaml` + `.env` + Pydantic `Settings` 统一配置体系（[src/config.py](https://github.com/lhy-0317/TCM-KGQA/blob/v0.1.0/src/config.py)）。
2. **统一大模型与向量编码抽象**：实现 `BaseLLM`（支持本地 4-bit `Qwen/Qwen3.5-4B` / `2B`、OpenAI/DashScope 兼容 API 与离线测试后端）和 `BaseEmbedder`（支持 `Qwen/Qwen3-Embedding-0.6B` MRL 截断、API、`SentenceTransformer` 与字符 n-gram TF-IDF）。
3. **中医数据工程与语义分块**：
   - 构建权威中医结构化核心库（[src/data/tcm_seed_data.py](https://github.com/lhy-0317/TCM-KGQA/blob/v0.1.0/src/data/tcm_seed_data.py)）：涵盖 **85 味核心中药**（含炮制品与别名）、**42 首经典名方**（含君臣佐使配伍及古籍出处）、完整 **十八反**、**十九畏** 与妊娠禁忌。
   - 接入 `ShenNong_TCM_Dataset` 流式下载与去重清洗、中文句界分块（`chunk_size=512, overlap=64`），共生成 **5,543 个有效知识分块**（包含 131 条药典/方剂/禁忌权威文档与 4,998 条清洗后的临床问答记录）。
4. **Vector RAG 基线与五维评测集**：
   - 实现支持 FAISS（含 NumPy 内积检索自动回退）的持久化 `VectorStore` 与 `BaselineRAGPipeline`。
   - 构建 52 题五维评测集 `data/eval/tcm_eval_benchmark.jsonl`（单跳药典、跨方剂多跳配伍禁忌、辨证选方、执业医师风格单选、域外拒识）。

---

## 二、v0.1 基线实验结果与问题分析

在 5,543 分块语料库与 52 题评测集上，基于本地 **`Qwen3.5-4B`（4-bit NF4）+ `Qwen3-Embedding-0.6B`** 测得基线表现如下：

| 系统配置 | Overall Acc (%) | 禁忌多跳 Acc (%) | 拒识 Acc (%) | Citation F1 (%) | 幻觉率 (%) | Recall@5 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| No-RAG (Base LLM) | 67.7 | 83.3 | 0.0 | 0.0 | 32.3 | — |
| v0.1 Vector-RAG Baseline | 88.9 | 83.9 | 100.0 | 91.3 | 9.1 | 89.4 |

**效果与暴露出的核心痛点（指导下一阶段优化）**：
1. **Vector-RAG 相对纯基座已有明显收益**：引入 `Qwen3-Embedding-0.6B` 检索后，综合准确率从 `67.7%` 提升至 `88.9%`，幻觉率从 `32.3%` 降至 `9.1%`，引用 F1 达到 `91.3%`。
2. **跨方剂多跳配伍禁忌仍有明显盲区**：面对“半夏泻心汤和四逆汤能否同服”等联合处方提问时，纯向量检索容易只召回单个方剂或漏掉对应的十八反规则文档（`Recall@5` 为 `89.4%`，`禁忌多跳 Acc` 停留在 `83.9%`，且无法给出显式 `[KG-j]` 可解释关系链）。
3. **炮制品别名与词界切分敏感度不足**：临床常见的“黑顺片”、“国老”、“法半夏”等炮制品别名及复合方名需要专门的中医实体链接与专词切分增强。

---

## 三、下一步计划

在下一轮迭代中，将针对上述痛点构建**中医多跳知识图谱（NetworkX/Neo4j 双后端）**、**三层实体链接消歧器**与 **十八反/十九畏跨方剂多跳推理引擎**，并引入 **中医专词 BM25 + Dense + KG 的 RRF 混合检索** 与精排模块。
