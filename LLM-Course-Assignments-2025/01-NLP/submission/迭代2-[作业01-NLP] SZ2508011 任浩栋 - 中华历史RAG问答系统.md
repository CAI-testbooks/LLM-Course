# 中国历史RAG问答系统

作为南航数学学院的一位研究生，我对中国的历史颇感兴趣，因此，我将构建一个基于Qwen大模型的RAG历史问答系统，并使用Lora微调提升领域准确性。
- **github仓库**：https://github.com/renhaodong1011/chinese-history-rag-qwen 。
- **目前已实现内容**：
- **迭代1**：1.编写了脚本model_download.py 用于下载Qwen-2.5-7B-Instruct模型到AutoDL本地；2.在文件夹data_extract中，编写脚本spider.py用于在中华上下五千年网站中爬取所有文本数据，共282个txt文 件；3.在文件夹data_extract中，编写脚本generate_QA_data.py,使用QWen4-8B模型，智能从爬取的所有数据中生成QA对，用于后续的Lora微调，共计提取QA对9978条，保存在相应文件夹下的chinese_history_qa.json文件中；4.在RAG.py文件中通过streamlit实现了一个完全本地化的中国历史领域 RAG（Retrieval-Augmented Generation）Web问答系统,支持流式输出。 且该第一版项目已经在AutoDL,A100-PCIE-40GB(40GB)单卡下部署成功。并将一些项目过程中的输出以截图的方式展示在了img文件夹中，包括数据浏览，QA对生成，RAG问答截图等。 后续的迭代，将以Lora微调模型为主线展开。
- **迭代2**： 1.新增Lora_tran.py 文件对Qwen-2.5-7B-Instruct模型进行Lora微调；2.新增merge_model.py 文件对微调后的模型进行合并； 3.将RAG中的模型替换为微调后的模型。 对搜集的9978条QA对进行微调在A100-PCIE-40GB(40GB)单卡模型下耗时2.5小时，一共训练了3个epoch, batch_size为4，结果显示Traning loss 和 Evaluation loss 曲线总体上呈现持续下降，Traning loss从一开始的2.9左右持续下降到0.5，Evaluation loss 从1.5左右持续下降到了0.3。最终在测试时，微调后RAG系统回答的更加精准以及简洁。

## 运行环境

- **平台** ： AutoDL(https://www.autodl.com/)。
- **镜像配置**： PyTorch==2.8.0 Python==3.12(ubuntu22.04) CUDA==12.8。
- **GPU** ： A100-PCIE-40GB(40GB)。

## 依赖安装

**执行命令**: pip install -r requirement.txt


## 数据来源

- **RAG知识库数据** : 用于RAG的检索模块，提供外部知识增强生成。 使用python爬虫从中华上下五千年网站（https://zhonghua.5000yan.com/）中，爬取所有的相关内容。
  **运行spider.py截图**：<img width="913" height="474" alt="3" src="https://github.com/user-attachments/assets/47838b0b-06fe-4aa9-8b7a-f12079ab6562" />
- **Lora微调数据获取** ： 使用现有的QWen3-8B模型，对每个历史文件提取并生成若干个QA对，用于LORA微调。
  **运行generate_QA_data.py获取QA对截图**：<img width="1920" height="1050" alt="10" src="https://github.com/user-attachments/assets/88dbac7b-f67c-427e-994d-5b2025ee6fa1" />

## 基础模型

使用 Qwen-2.5-7B-instruct 为基础模型
在AutoDL中，在 /root/autodl-tmp 路径下新建 model_download.py 用于下载完整的模型。
``` Python
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('qwen/Qwen2.5-7B-Instruct', cache_dir='/root/autodl-tmp', revision='master')
```
![模型目录] <img width="245" height="412" alt="1" src="https://github.com/user-attachments/assets/cbc17021-6fa0-4fcd-9284-7a0b5c74d6b0" />

## 文件作用介绍
- **1.model_download.py**:用于下载Qwen-2.5-7B-Instruct模型到AutoDL本地。
- **2.data_extract/spider.py**: 用于在中华上下五千年网站中爬取所有文本数据，共282个txt文件。
- **3.data_extract/generate_QA_data.py**:使用QWen4-8B模型，智能从爬取的所有数据中生成QA对，用于后续的Lora微调，共计提取QA对9978条，保存在相应文件夹下的chinese_history_qa.json文件中。
- **4.RAG.py**: 通过streamlit实现了一个完全本地化的中国历史领域 RAG（Retrieval-Augmented Generation）Web问答系统,支持流式输出。
- **5.Lora_train**: 对Qwen-2.5-7B-Instruct模型进行Lora微调。
- **6.merge_model**: 对微调后的模型进行合并

## 启动方式

streamlit run RAG.py

## 运行截图（第一版，未微调）
<img width="1912" height="972" alt="6" src="https://github.com/user-attachments/assets/c5db1c26-2f3a-4c7c-a515-7e9d235b7a52" />
<img width="1912" height="972" alt="7" src="https://github.com/user-attachments/assets/9f3cdd56-7c10-4a87-9949-b31f412c98e2" />
<img width="1912" height="972" alt="8" src="https://github.com/user-attachments/assets/ffa8f600-aea7-48d3-ac39-5a2121e2f63e" />
<img width="1912" height="972" alt="9" src="https://github.com/user-attachments/assets/1322e128-cc37-4ec0-8f2a-fc86bdb6dfc5" />

## Lora 微调
- **训练过程和结果** ：
  <img width="1553" height="654" alt="12" src="https://github.com/user-attachments/assets/224f36e7-9cdc-489d-be7e-85110d3cbe59" />
  <img width="1222" height="504" alt="11" src="https://github.com/user-attachments/assets/d3bc4b58-ff10-4860-80e3-554431a6d5a4" />
  <img width="552" height="391" alt="13" src="https://github.com/user-attachments/assets/ceb9b62f-e5e3-46a7-bfe9-20c6480db1f4" />

- **训练后RAG运行截图**：
  <img width="1920" height="1050" alt="14" src="https://github.com/user-attachments/assets/39a801a6-db85-43e6-967d-f79adea4f746" />
  <img width="1920" height="1050" alt="15" src="https://github.com/user-attachments/assets/e0b3e539-9bad-4065-b03f-385c352d9e4f" />
