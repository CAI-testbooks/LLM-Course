本次作业所需代码：https://github.com/chen-kkk/NLP_HW_code
本次作业所需环境：transformers>=4.41.2,<=4.45.2
datasets>=2.16.0,<=2.21.0
accelerate>=0.30.1,<=0.34.2
peft>=0.11.1,<=0.12.0
trl>=0.8.6,<=0.9.6
gradio>=4.0.0
pandas>=2.0.0
scipy
einops
sentencepiece
tiktoken
protobuf
uvicorn
pydantic
fastapi
sse-starlette
matplotlib>=3.7.0
fire
packaging
pyyaml
numpy<2.0.0


步骤很简单，用我的数据集微调模型，修改后的LLaMA-Factory可以实现上下文多轮对话
模型文件可自行下载
评测、微调、数据相关代码均在链接中，难度不大，极易复现

bash /root/LLaMA-Factory/chuli/one.sh 启动web界面