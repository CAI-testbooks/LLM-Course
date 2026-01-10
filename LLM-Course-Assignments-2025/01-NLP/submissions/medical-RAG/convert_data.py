import json
import os
from datasets import load_dataset
from tqdm import tqdm

# 1. 配置路径
LOCAL_DATASET_PATH = "./Huatuo26M-Lite" 
OUTPUT_FILE = "medical_sft_data.json"
MAX_SAMPLES = 10000

def convert_to_alpaca():
    print(f"📂 正在读取本地数据集: {LOCAL_DATASET_PATH}...")
    try:
        # 加载本地数据集
        dataset = load_dataset(LOCAL_DATASET_PATH, split="train")
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    sft_data = []
    print("🔄 正在转换格式为 Alpaca 指令集...")
    
    for i, item in tqdm(enumerate(dataset), total=min(len(dataset), MAX_SAMPLES)):
        if i >= MAX_SAMPLES:
            break
        if not item.get('question') or not item.get('answer'):
            continue
            
        entry = {
            "instruction": "你是一名专业的医疗智能助手。请根据用户的问题，给出准确、专业且客观的医疗建议。",
            "input": item['question'],
            "output": item['answer']
        }
        sft_data.append(entry)

    # 保存文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)
        
    print(f"✅ 转换完成！已保存 {len(sft_data)} 条微调数据至 {OUTPUT_FILE}")

if __name__ == "__main__":
    convert_to_alpaca()