# 大模型原理与技术作业

* BX2524305-08-Others-基于认知和MRI对齐的视觉语言预训练脑疾病诊断模型


## 08-Others
* This is for submitting LMM course assignments.


## Run
Prepare the necessary brain imaging data.

* "python -m torch.distributed.run --nproc_per_node=8 run_homework_SMRI_BLIP2_train.py"

* and "python -m torch.distributed.run --nproc_per_node=8 run_homework_SMRI_BLIP2_test.py"

## Requirements
* The following setup has been tested on Python 3.9, Ubuntu 20.04. 

* Major dependences: pytorch 1.13.1 salesforce-lavis 1.0.2 transformers 4.28.1

