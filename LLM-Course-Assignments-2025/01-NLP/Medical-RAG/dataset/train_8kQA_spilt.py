import json
import os

# 配置路径
input_file = "/root/autodl-tmp/Medical-RAG/dataset/train_data.json"
output_file = "/root/autodl-tmp/Medical-RAG/dataset/train_data_8k.json"
num_samples = 8000

def extract_first_n_lines(input_path, output_path, n):
    if not os.path.exists(input_path):
        print(f"❌ 输入文件不存在: {input_path}")
        return

    count = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue  # 跳过空行

            try:
                # 验证是否为合法 JSON（可选，确保数据干净）
                data = json.loads(line)
                # 写回原样（不修改内容）
                fout.write(line + "\n")
                count += 1
                if count >= n:
                    break
            except json.JSONDecodeError:
                print(f"⚠️ 跳过无效 JSON 行: {line[:100]}...")

    print(f"✅ 成功提取 {count} 条 QA 对 到 {output_file}")

if __name__ == "__main__":
    extract_first_n_lines(input_file, output_file, num_samples)