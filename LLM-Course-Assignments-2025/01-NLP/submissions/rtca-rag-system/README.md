# RTCA DO-160G RAG智能问答系统

基于Qwen-2.5的航空标准文档智能问答系统，支持多轮对话、引用显示和不确定性检测。

## 特性

- ✅ 支持RTCA DO-160G标准文档问答
- ✅ 混合检索策略（稠密+稀疏+重排序）
- ✅ 多轮对话上下文管理
- ✅ 引用来源显示和验证
- ✅ 不确定性检测和拒绝回答
- ✅ LoRA/SFT微调支持
- ✅ 多种部署方式（Web/API）
- ✅ 完整的评估指标体系

## 快速开始

### 1. 环境安装

```bash

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt