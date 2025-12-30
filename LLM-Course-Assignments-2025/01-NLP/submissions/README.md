# MedQA-RAG: A Medical QA System Powered by DeepSeek-R1

智疗问答：基于 DeepSeek-R1 的医疗 RAG 系统

本项目实现了一个**中文医疗领域特定问答系统（Domain-Specific QA）**：

* 使用 **cMedQA2** 数据集构建 QA 知识库，并通过 **Chroma** 持久化为向量库
* 基于 **Ollama 本地大模型（deepseek-r1:8b）+ Embedding 模型** 实现 RAG（支持 Ollama Embedding / 本地 HF Embedding）
* 支持 **Embedding 模型微调（SFT / 可选 LoRA）**：只微调检索端向量模型，不进行 LLM 微调
* 提供 **评估脚本**（准确率、幻觉率）与 **Streamlit Web Demo**（多轮对话、引用来源展示、拒绝不确定回答、长上下文策略）

> 向量库构建脚本会将「问题 + 正确回答」拼成文档并写入 Chroma 向量库。
> Web Demo 使用本地 `ollama:deepseek-r1:8b` 作为对话模型，并加载 `./chroma_rag_db` 向量库进行检索增强。

---

## 功能特性

* **RAG 检索增强问答**：从 Chroma 向量库检索相关 QA 资料拼接为上下文后回答
* **多轮对话**：保留对话历史，并可选进行“查询改写（query rewrite）”以提升多轮检索效果
* **长上下文策略**：对话历史使用“滚动摘要（summary）+ 最近 N 轮窗口”的方式支持长对话（避免输入无限增长）
* **引用来源显示**：UI 展示每轮检索到的资料条目（包含 answer_id / question_id / chunk_id / score），并在回答中输出引用来源区块
* **拒绝不确定回答**：当检索相似度不足（距离阈值过大）时触发拒答，并给出保守建议与补充信息提示
* **Embedding 方案可切换**：构建向量库时可选择

  * Ollama Embedding（如 `qwen3-embedding:4b`）
  * 本地 Hugging Face Embedding（可使用你微调后的 embedding 模型）
* **评估**：提供 Baseline（不使用向量库）与 RAG 版本评估流程，输出准确率与幻觉率，支持进度条与样本数限制

---

## 数据集获取

* cMedQA2：`https://github.com/zhangsheng93/cMedQA2`

数据格式示例：

* `question.csv`: `question_id, content`
* `answer.csv`: `ans_id, question_id, content`
* `train_candidates.txt / dev_candidates.txt / test_candidates.txt`：候选对（测试集只取 `label=1` 的样本用于评估）

---

## 环境准备（Conda + Python 3.11）

1. 创建并进入 conda 环境：

```bash
conda create -n medqa-rag python=3.11 -y
conda activate medqa-rag
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 安装并启动 Ollama（本地推理服务）

* 确保本地可访问 `http://localhost:11434`（项目默认使用此地址）
* 需要提前拉取/准备：

  * `deepseek-r1:8b`（对话模型）
  * `qwen3-embedding:4b`（Embedding 模型，若你使用 Ollama Embedding）

4)（可选）安装 embedding 微调相关依赖（若你需要微调向量模型）

```bash
pip install sentence-transformers torch
# 可选：LoRA 适配器训练
pip install peft
```

---

## 配置说明（config.py）

编辑 `config.py`，将数据集路径改为你本机路径（示例为 Windows 绝对路径）：

* `question_dict / answer_dict / train_dict / dev_dict / test_dict`：数据文件路径
* `samples_num`：评估采样数量（如 100；设为 `None` 表示全量）

---

## 1）Embedding 模型微调（可选：SFT / LoRA）

> 说明：这里仅微调检索端 embedding 模型（训练数据来自 cMedQA2 的问答对），**不对 LLM 做微调**。
> 适用场景：你希望“医疗问答语义匹配”更贴合数据集，从而提升检索质量与最终回答效果。

训练本地 embedding 模型（输出到 `models/medqa-embedding`）：

```bash
python finetune_embedding.py --data_dir dataset/cMedQA2 --output_dir models/medqa-embedding
```

可选：使用 LoRA（只训练小型适配器，更轻量）：

```bash
python finetune_embedding.py --data_dir dataset/cMedQA2 --output_dir models/medqa-embedding --use_lora --lora_target query,key,value
```

---

## 2）构建向量库（build_vector_store.py）

该脚本会：

* 从训练集候选对中提取 `(question_id, pos_ans_id)`
* 拼接为文档：`问题：... \n\n回答：...`
* 分块（500 字符、100 overlap，中文分隔符优先）
* 写入 Chroma 并持久化到 `./chroma_rag_db`

### 方案 A：使用 Ollama Embedding（默认）

```bash
python build_vector_store.py --embedding-backend ollama --ollama-model qwen3-embedding:4b --ollama-base-url http://localhost:11434
```

### 方案 B：使用本地 Hugging Face embedding（例如你微调后的模型）

```bash
python build_vector_store.py --embedding-backend hf --hf-model-path models/medqa-embedding
```

输出目录：

* `./chroma_rag_db`（Chroma 持久化向量库）

---

## 3）评估（eval.py）

该脚本会分别评估：

* **Baseline**：不使用向量库，直接调用模型回答，并用 LLM-as-judge 计算准确率/幻觉率
* **RAG**：加载向量库检索后再回答，并同样计算指标

运行：

```bash
python eval.py
```

评估输出示例：

```json
{
  "n": 100,
  "accuracy": 0.87,
  "hallucination_rate": 0.05
}
```

> 说明：`n` 是参与评测的样本数；`accuracy` 是 judge 判为正确的比例；`hallucination_rate` 是 judge 判为幻觉的比例。

---

## 4）启动 Web Demo（app.py）

本项目提供 Streamlit 可视化聊天界面，支持：

* 多轮对话（历史消息 + 滚动摘要）
* RAG 检索增强（Top-K 可调）
* 引用来源显示（展示每轮检索到的资料与 metadata）
* 拒答机制（基于检索距离阈值）

运行：

```bash
streamlit run app.py
```

侧边栏参数说明：

* `Top-K`：每轮检索返回的文档条数
* `拒答距离阈值`：阈值越小越严格；当最优检索距离仍偏大时拒答
* `保留最近对话轮数`：窗口记忆大小（配合 summary 实现长对话）
* `启用查询改写`：多轮场景下提升检索命中（可开关）

---

## 项目结构

* `build_vector_store.py`：构建向量库（支持 Ollama / HF embedding，Chroma 持久化到 `./chroma_rag_db`）
* `finetune_embedding.py`：Embedding 微调脚本（SFT，可选 LoRA）
* `eval.py`：评估 Baseline vs RAG（准确率、幻觉率，含进度条）
* `app.py`：Streamlit Web Demo（多轮、长上下文策略、引用来源、拒答）
* `config.py`：数据路径与评估参数配置
* `requirements.txt`：依赖列表（pip 安装）

---

## 常见问题（Troubleshooting）

1. **向量库为空 / app 检索不到内容**

* 确认已先运行构建向量库脚本生成 `./chroma_rag_db`
* 确认 `config.py` 中数据路径正确（尤其是 Windows 绝对路径）
* 若使用 HF embedding：确认 `--hf-model-path` 指向的模型目录存在且可加载

2. **Ollama 连接失败**

* 确认 Ollama 服务已启动且可访问 `http://localhost:11434`
* 若使用 Ollama embedding：确认已拉取模型 `qwen3-embedding:4b`
* 确认对话模型 `deepseek-r1:8b` 已就绪

3. **拒答太频繁 / 太少**

* 在 app 侧边栏调整“拒答距离阈值”（越小越严格；越大越宽松）

4. **是否一定要微调 embedding？**

* 不必须：你可以直接用 `qwen3-embedding:4b` 构建向量库
* 但在医疗问答这种强领域语义匹配任务中，embedding 微调常常能明显改善检索命中，从而减少幻觉、提升准确率

---

## 免责声明

本系统仅用于课程作业与科研实验演示，**不构成医疗诊断或治疗建议**。如遇严重或紧急症状，请及时就医。

---

## 致谢

* 数据集：cMedQA2（见上方仓库链接）
* 向量库：Chroma
* 推理与嵌入：Ollama 本地模型服务

