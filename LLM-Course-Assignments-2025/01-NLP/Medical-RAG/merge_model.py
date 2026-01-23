#merge  代码
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model_path = "/root/autodl-tmp/qwen/Qwen2___5-7B-Instruct"
lora_path = "/root/autodl-tmp/Medical-RAG/Tune-model/medical-qwen-lora/checkpoint-2700"
output_path = "/root/autodl-tmp/Medical-RAG/Tune-model/medical-qwen-merged"
#还是不改变分词器，只改变权重即可，加载我们的checkpoint的最后一个即可。
#尽管我们的微调的数据过拟合，没办法，数据集本身的量小，与显卡的限制不适合大规模的微调。
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(model, lora_path)
model = model.merge_and_unload()

model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print(f"✅ 合并完成！模型已保存至: {output_path}")