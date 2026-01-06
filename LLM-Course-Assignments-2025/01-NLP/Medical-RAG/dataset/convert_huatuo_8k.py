import json

def convert_huatuo_to_alpaca(input_file_path, output_file_path):
    """
    将华佗数据集 (huatuo_encyclopedia_qa) 转换为 Alpaca 格式。

    Args:
        input_file_path (str): 原始 train_data_8k.json 文件路径。
        output_file_path (str): 输出的 Alpaca 格式 JSON 文件路径。
    """
    alpaca_data = []

    with open(input_file_path, 'r', encoding='utf-8') as infile:
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue # 跳过空行
            
            try:
                # 解析每一行的独立 JSON 对象
                data_record = json.loads(line)
                questions = data_record.get("questions", [])
                answers = data_record.get("answers", [])
                
                # 安全检查：确保有且仅有一个答案
                if len(answers) != 1:
                    print(f"警告: 第 {line_num} 行的答案数量不是1 ({len(answers)})，跳过此记录。")
                    continue
                
                answer_text = answers[0]
                
                # 为每个问题创建一个独立的 QA 对
                for question_list in questions:
                    for question in question_list:
                        alpaca_entry = {
                            "instruction": question,
                            "input": "",
                            "output": answer_text
                        }
                        alpaca_data.append(alpaca_entry)
            
            except json.JSONDecodeError as e:
                print(f"错误: 第 {line_num} 行 JSON 解析失败: {e}")
                continue

    # 将最终的 Alpaca 数据写入新文件
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        json.dump(alpaca_data, outfile, ensure_ascii=False, indent=2)
    
    print(f"转换完成！共生成 {len(alpaca_data)} 个 QA 对。")
    print(f"输出文件已保存至: {output_file_path}")

if __name__ == "__main__":
    # 请根据您的实际文件路径修改以下变量
    train_INPUT_FILE = "/root/autodl-tmp/Medical-RAG/dataset/train_data_8k.json"
    train_OUTPUT_FILE = "/root/autodl-tmp/Medical-RAG/dataset/alpaca_formatted_train_data_8k.json"
    test_INPUT_FILE = "/root/autodl-tmp/Medical-RAG/dataset/test_data.json"
    test_OUTPUT_FILE = "/root/autodl-tmp/Medical-RAG/dataset/alpaca_formatted_test_data.json"
    validation_INPUT_FILE = "/root/autodl-tmp/Medical-RAG/dataset/validation_data.json"
    validation_OUTPUT_FILE = "/root/autodl-tmp/Medical-RAG/dataset/alpaca_formatted_validation_data.json"
    convert_huatuo_to_alpaca(train_INPUT_FILE, train_OUTPUT_FILE)
    convert_huatuo_to_alpaca(test_INPUT_FILE, test_OUTPUT_FILE)
    convert_huatuo_to_alpaca(validation_INPUT_FILE, validation_OUTPUT_FILE)
    