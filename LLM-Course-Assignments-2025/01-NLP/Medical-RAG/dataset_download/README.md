---
license: apache-2.0
task_categories:
- text-generation
language:
- zh
tags:
- medical
size_categories:
- 100K<n<1M
---
# Dataset Card for Huatuo_encyclopedia_qa



## Dataset Description

- **Homepage: https://www.huatuogpt.cn/** 
- **Repository: https://github.com/FreedomIntelligence/HuatuoGPT** 
- **Paper: https://arxiv.org/abs/2305.01526** 
- **Leaderboard:** 
- **Point of Contact:** 



### Dataset Summary
百科问答数据集
本数据集共包含 364,420 条中文医疗问答（QA）数据，其中部分条目以不同方式提出了多个问题。我们从纯文本资源（如医学百科全书和医学文章）中提取了这些医疗问答对。具体而言，我们收集了中文维基百科上 8,699 篇疾病类百科条目 和 2,736 篇药品类百科条目，此外还从“千问健康”网站爬取了 226,432 篇高质量医学文章。



## Dataset Creation



### Source Data

https://zh.wikipedia.org/wiki/  

https://51zyzy.com/




## Citation

```
@misc{li2023huatuo26m,
      title={Huatuo-26M, a Large-scale Chinese Medical QA Dataset}, 
      author={Jianquan Li and Xidong Wang and Xiangbo Wu and Zhiyi Zhang and Xiaolong Xu and Jie Fu and Prayag Tiwari and Xiang Wan and Benyou Wang},
      year={2023},
      eprint={2305.01526},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
