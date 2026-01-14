## 项目描述

构建一个基于Qwen-2.5-7B-Instruct模型的中文医疗领域智能问答系统。选择一个医疗专业领域，收集医疗领域中华佗数据集，实现一个支持32K长上下文、多轮对话的RAG（Retrieval-Augmented Generation）问答系统，能够准确回答专业的医疗领域的相关问题。

### 项目信息

- **源代码路径**：https://github.com/Joshua00044444/LLM-Course/tree/feature/NLP-MedicalRAG/LLM-Course-Assignments-2025/01-NLP
- **选择分支**：feature/NLP-MedicalRAG

### 复现说明

所有命令在租用的显卡目录下运行即可：`/root/autodl-tmp`

---

## 项目分析

### 1. 模型选择

首先下载Qwen-2.5-7B-Instruct模型至本地。

运行命令：
```bash
python model_download/download-qwen2.5-7b-instruct.py
```

输出结果：模型会下载至 `/root/autodl-tmp/qwen/qwen2.5-7b-instruct`

**模型下载截图：**

![模型下载1](/LLM-Course-Assignments-2025/01-NLP/Medical-RAG/img/image1.png)

![模型下载2](/LLM-Course-Assignments-2025/01-NLP/Medical-RAG/img/image2.png)

---

### 2. 数据准备

收集至少8K条医疗领域QA数据，使用 **huatuo_encyclopedia_qa** 百科问答数据集。

**数据集介绍：**

本数据集共包含 **364,420 条**中文医疗问答（QA）数据，其中部分条目以不同方式提出了多个问题。我们从纯文本资源（如医学百科全书和医学文章）中提取了这些医疗问答对。

具体而言，我们收集了：
- 中文维基百科上 **8,699 篇**疾病类百科条目
- **2,736 篇**药品类百科条目
- 此外还从"千问健康"网站爬取了 **226,432 篇**高质量医学文章

从中挑选 **8K 条** QA对数据，针对数据进行清洗与转换为Alpaca格式。并构建持久化向量数据库，用于后续的检索。

**数据处理截图：**

数据集下载与提取8K条QA对：

![数据集下载与提取8K条QA对](/LLM-Course-Assignments-2025/01-NLP/Medical-RAG/img/image3.png)

数据集清洗与处理后：

![数据集清洗与处理后](/LLM-Course-Assignments-2025/01-NLP/Medical-RAG/img/image4.png)

数据集构建持久化向量数据库：

![数据集构建持久化向量数据库](/LLM-Course-Assignments-2025/01-NLP/Medical-RAG/img/image5.png)

---

### 3. 部署

使用Streamlit构建Web demo，支持实时交互。

**基于Streamlit的RAG流式问答系统构建：**

安装关键库：

```bash
pip install langchain langchain-community langchain-openai chromadb python-dotenv streamlit

python -m streamlit run Medical-RAG/Medical-RAG.py 
```

**功能特性：**
- 支持多轮对话
- 长上下文（>32k tokens）
- 拒绝不确定回答

**部署运行截图：**

![部署截图1](/LLM-Course-Assignments-2025/01-NLP/Medical-RAG/img/image6.png)

![部署截图2](/LLM-Course-Assignments-2025/01-NLP/Medical-RAG/img/image7.png)

![部署截图3](/LLM-Course-Assignments-2025/01-NLP/Medical-RAG/img/image8.png)

---

### 4. 迭代优化

实施 **MMR检索策略**、优化 **上下文prompt**，使用 **BGE-Reranker-v2-m3** 重排序机制。

#### 优化前：

![优化前截图1](/LLM-Course-Assignments-2025/01-NLP/Medical-RAG/img/image9.png)

![优化前截图2](/LLM-Course-Assignments-2025/01-NLP/Medical-RAG/img/image10.png)

![优化前截图3](/LLM-Course-Assignments-2025/01-NLP/Medical-RAG/img/image11.png)

![优化前截图4](/LLM-Course-Assignments-2025/01-NLP/Medical-RAG/img/image12.png)

#### 优化后：

![优化后截图1](/LLM-Course-Assignments-2025/01-NLP/Medical-RAG/img/image13.png)

![优化后截图2](/LLM-Course-Assignments-2025/01-NLP/Medical-RAG/img/image14.png)

![优化后截图3](/LLM-Course-Assignments-2025/01-NLP/Medical-RAG/img/image15.png)

![优化后截图4](/LLM-Course-Assignments-2025/01-NLP/Medical-RAG/img/image16.png)

#### 指标对比

使用准确率、引用F1、幻觉率等指标，比较基线与RAG版本。

![指标对比1](/LLM-Course-Assignments-2025/01-NLP/Medical-RAG/img/image17.png)

---

### 5. 模型选择与微调

基于开源LLM **Qwen-2.5**，使用 **LoRA微调** 提升领域准确率，使模型理解数据集的生成逻辑能力。

#### 微调过程：

![微调结果1](/LLM-Course-Assignments-2025/01-NLP/Medical-RAG/img/image18.png)

![微调结果2](/LLM-Course-Assignments-2025/01-NLP/Medical-RAG/img/image19.png)

> **注：** 8K数据集还是有点小了，没办法，算力有限，实在租不起显卡了。

#### 微调后对比

微调后合并的模型引入RAG优化后，与未微调RAG优化后的模型对比：

**微调后的Streamlit界面：**

![微调后的streamlit界面](/LLM-Course-Assignments-2025/01-NLP/Medical-RAG/img/image20.png)

**微调后合并模型：**

![微调后合并模型](/LLM-Course-Assignments-2025/01-NLP/Medical-RAG/img/image21.png)

**模型替换：**

![模型替换](/LLM-Course-Assignments-2025/01-NLP/Medical-RAG/img/image22.png)

**未微调RAG优化后：**

![未微调RAG优化后](/LLM-Course-Assignments-2025/01-NLP/Medical-RAG/img/image23.png)

**微调后合并的模型引入RAG优化：**

![微调后合并的模型引入RAG优化](/LLM-Course-Assignments-2025/01-NLP/Medical-RAG/img/image24.png)

---

## 总结

本项目成功实现了基于Qwen-2.5-7B-Instruct的中文医疗领域智能问答系统，通过RAG技术和LoRA微调相结合，在有限的算力下构建了一个功能完善的医疗问答平台。
