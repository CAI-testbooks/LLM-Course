## 题目描述

构建一个基于Qwen-2.5-7B-Instruct模型的中文医疗领域智能问答系统。选择一个医疗专业领域，收集医疗领域中华佗数据集，实现一个支持32K长上下文、多轮对话的RAG（Retrieval-Augmented Generation）问答系统，能够准确回答专业的医疗领域的相关问题。

## 题目要求
- **模型选择**
(所有命令在/root/autodl-tmp下运行即可)
首先先下载Qwen-2.5-7B-Instruct模型至本地，运行如下命令：python  model_download/download-qwen2.5-7b-instruct.py

![模型下载1](/LLM-Course/LLM-Course-Assignments-2025/01-NLP/Medical-RAG/img/image1.png)

![模型下载2](/LLM-Course/LLM-Course-Assignments-2025/01-NLP/Medical-RAG/img/image2.png)



- **数据准备**：收集至少5k条领域QA对或文档（可使用公开数据集+爬取/合成），进行清洗、分块、构建向量数据库。
- **模型选择与微调**：基于开源LLM（如Qwen-2.5、LLaMA-3.1或Gemma-2）实现RAG，可选LoRA/SFT微调提升领域准确率。
- **核心功能**：支持多轮对话、长上下文（>32k tokens）、引用来源显示、拒绝不确定回答。
- **迭代优化**：全程可更新embedding模型、检索策略、prompt或微调数据，提升在领域基准（如CMedQA、LegalBench）上的准确率。
- **部署**：使用Gradio/Streamlit/FastAPI构建Web demo，支持实时交互。
- **评估**：使用准确率、引用F1、幻觉率等指标，比较基线与优化后版本。


