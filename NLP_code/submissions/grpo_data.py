import json

# 定义输入和输出文件路径
input_file = '/root/autodl-tmp/train.json'
output_file = '/root/autodl-tmp/train_converted.json'

# 读取原始数据集
with open(input_file, 'r', encoding='utf-8') as f:
    original_data = json.load(f)

# 转换数据格式
converted_data = []
for item in original_data:
    new_item = {
        'prompt': [
            {'role': 'system', 'content': item['instruction']},
            {'role': 'user', 'content': item['input']}
        ],
        'answer': item['output']
    }
    converted_data.append(new_item)

# 保存转换后的数据
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(converted_data, f, ensure_ascii=False, indent=2)

print(f"数据转换完成！共处理 {len(converted_data)} 条记录")
print(f"转换后的文件已保存至: {output_file}")