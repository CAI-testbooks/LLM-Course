import os
import json
import requests
import time

# === 配置 ===
# 使用国内镜像站下载，速度快且稳定
DATA_URL = "https://hf-mirror.com/datasets/shibing624/medical/resolve/main/finetune/train_zh_0.json"
SAVE_PATH = "../data/medical.json"
TARGET_COUNT = 8000  # 我们目标是取 8000 条，满足作业 5k+ 的要求


def download_and_process():
    print(f"🚀 开始下载真实医疗数据集...")
    print(f"🔗 下载源: {DATA_URL}")

    try:
        # 1. 流式下载（防止文件过大撑爆内存）
        response = requests.get(DATA_URL, stream=True, timeout=60)
        response.raise_for_status()

        raw_lines = []
        count = 0

        # 2. 逐行读取并处理
        # 这个数据集原本是 jsonl 格式（每一行是一个独立的 json 对象）
        for line in response.iter_lines():
            if not line:
                continue

            if count >= TARGET_COUNT:
                break

            try:
                item = json.loads(line.decode('utf-8'))
                # 提取需要的字段
                # 原始数据的字段通常是 input/output/instruction
                q = item.get("instruction", "") + item.get("input", "")
                a = item.get("output", "")

                if q and a:
                    # 重新构造成我们需要格式
                    raw_lines.append({
                        "instruction": q,
                        "output": a
                    })
                    count += 1

                    if count % 1000 == 0:
                        print(f"   已处理 {count} 条数据...")

            except json.JSONDecodeError:
                continue

        # 3. 保存为我们要的 JSON 格式
        if not os.path.exists("../data"):
            os.makedirs("../data")

        print(f"💾 正在保存到 {SAVE_PATH}...")
        with open(SAVE_PATH, "w", encoding="utf-8") as f:
            json.dump(raw_lines, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 40)
        print(f"✅ 成功！已生成真实的医疗知识库")
        print(f"📊 数据量: {len(raw_lines)} 条 (满足作业 >5k 要求)")
        print(f"📂 文件位置: {os.path.abspath(SAVE_PATH)}")
        print("=" * 40)
        print("👉 下一步：请重启你的 main.py，系统会自动加载这些新数据！")

    except Exception as e:
        print(f"\n❌ 下载或处理失败: {str(e)}")
        print("建议：如果脚本下载失败，请手动访问上面的 URL 下载文件，并重命名为 data/medical.json")


if __name__ == "__main__":
    download_and_process()