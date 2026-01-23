import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset

# 配置
DATASET_NAME = "FreedomIntelligence/Huatuo26M-Lite"
SAVE_PATH = "./huatuo_local_data"

def download_and_save():
    print(f"🚀 开始下载数据集: {DATASET_NAME} ...")

    try:
        dataset = load_dataset(DATASET_NAME, split="train")
        
        print(f"✅ 下载完成！共 {len(dataset)} 条数据。")
        print(f"💾 正在保存到本地磁盘: {SAVE_PATH} ...")
        
        dataset.save_to_disk(SAVE_PATH)
        
        print(f"🎉 成功！数据集已保存至 {SAVE_PATH}，下一步可直接读取。")
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")

if __name__ == "__main__":
    download_and_save()