import os
from datasets import load_dataset

# 只加载百科问答数据集
encyclopedia_dataset = load_dataset('FreedomIntelligence/huatuo_encyclopedia_qa')

print("数据集分割:", list(encyclopedia_dataset.keys()))
print("训练集样本数:", len(encyclopedia_dataset['train']))
print("验证集样本数:", len(encyclopedia_dataset['validation']))
print("测试集样本数:", len(encyclopedia_dataset['test']))

# 保存数据集到指定目录
save_dir = "Medical-RAG/dataset"  # 目标目录
os.makedirs(save_dir, exist_ok=True)

# 保存整个数据集
try:
    # 保存为本地文件
    encyclopedia_dataset.save_to_disk(save_dir)
    print(f"数据集已成功保存到: {save_dir}")
    
    # 同时保存为JSON格式便于查看
    for split in ['train', 'validation', 'test']:
        json_path = os.path.join(save_dir, f"{split}_data.json")
        encyclopedia_dataset[split].to_json(json_path, orient='records', lines=True, force_ascii=False)
        print(f"{split}集已保存为JSON: {json_path}")
        
except Exception as e:
    print(f"保存数据集时出错: {e}")
    # 如果保存失败，尝试其他格式
    for split in ['train', 'validation', 'test']:
        try:
            csv_path = os.path.join(save_dir, f"{split}_data.csv")
            encyclopedia_dataset[split].to_csv(csv_path, index=False)
            print(f"{split}集已保存为CSV: {csv_path}")
        except Exception as csv_error:
            print(f"保存{split}集为CSV失败: {csv_error}")
