# train_data_8k.py - 提取前 8000 行 JSONL 数据
input_file = "/root/autodl-tmp/Medical-RAG/dataset/train_data.json"
output_file = "/root/autodl-tmp/Medical-RAG/dataset/train_data_8k.json"
n_lines = 8000

print(f"正在从 {input_file} 读取前 {n_lines} 行（JSONL 格式）...")

with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:

    count = 0
    for line in fin:
        if count >= n_lines:
            break
        # 可选：验证是否为有效 JSON（跳过空行）
        line = line.strip()
        if not line:
            continue
        fout.write(line + "\n")
        count += 1

print(f"✅ 成功提取 {count} 行，保存至: {output_file}")