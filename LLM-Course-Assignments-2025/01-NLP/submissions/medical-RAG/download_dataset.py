import os

# 1. 优先设置镜像环境变量 (必须在 import datasets 之前)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset

# 配置
DATASET_NAME = "FreedomIntelligence/Huatuo26M-Lite"
SAVE_PATH = "./huatuo_local_data"  # 数据集将保存在这个文件夹

def download_and_save():
    print(f"🚀 开始下载数据集: {DATASET_NAME} ...")
    print("注意：如果数据集较大，第一次下载可能需要几分钟。")

    try:
        # 下载数据集 (不使用 streaming，直接下载全部)
        # split="train" 表示只下载训练集
        dataset = load_dataset(DATASET_NAME, split="train")
        
        print(f"✅ 下载完成！共 {len(dataset)} 条数据。")
        print(f"💾 正在保存到本地磁盘: {SAVE_PATH} ...")
        
        # 保存到本地磁盘
        dataset.save_to_disk(SAVE_PATH)
        
        print(f"🎉 成功！数据集已保存至 {SAVE_PATH}，下一步可直接读取。")
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")

if __name__ == "__main__":
    download_and_save()