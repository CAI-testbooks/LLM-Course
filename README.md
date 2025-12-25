# 自动控制原理智能助教 RAG 系统

## 1. 项目简介
本项目是一款针对“自动控制原理”课程开发的深度 RAG（检索增强生成）问答系统。系统以陈复扬《自动控制原理》及习题集为核心知识源，利用 Qwen-2.5 大模型实现专业知识的精准检索与智能化解答。

## 2. 核心功能
* **大规模专业语料**：基于 1247 条原始片段合成了 5000+ 条高质量 QA 对。
* **专业数学支持**：完美支持 LaTeX 公式渲染，确保传递函数 $G(s)$ 等公式清晰显示。
* **高性能检索**：采用 FAISS 本地向量数据库，实现毫秒级知识定位。
* **拒绝虚假回答**：内置阈值过滤逻辑，有效抑制模型幻觉。

## 3. 代码结构说明
* `step01_process_data.py`: 原始教材语料清洗与分块
* `step02_build_db.py`: 构建 FAISS 向量数据库
* `step04_rag_cloud.py`: 云端 RAG 链路测试
* `step05_data_augmentation_cloud.py`: 大规模数据增强生产脚本

## 4. 环境安装与启动
### 依赖安装
```bash
pip install langchain transformers faiss-cpu sentence-transformers
