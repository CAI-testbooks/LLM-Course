import json
import os

# ================= 配置路径 =================
INPUT_FILE = "final_control_data_5050.json"
OUTPUT_FILE = "control_sft_data.jsonl"


# ===========================================

def extract_text(field_content):
    """
    高级容错：处理嵌套的字典或列表内容
    """
    if isinstance(field_content, dict):
        # 提取题干 (q)
        q = field_content.get('q', '')

        # 提取选项 (options)
        options = field_content.get('options', [])
        processed_options = []

        if isinstance(options, list):
            for opt in options:
                # 核心修复：如果选项本身还是列表，将其打平为字符串
                if isinstance(opt, list):
                    processed_options.append(" ".join([str(i) for i in opt]))
                else:
                    processed_options.append(str(opt))

        options_str = "\n".join(processed_options)
        return f"{q}\n{options_str}".strip()

    # 如果已经是字符串或其他基本类型，直接返回
    return str(field_content)


def convert_format():
    # 确保在脚本所在目录下寻找文件
    if not os.path.exists(INPUT_FILE):
        print(f"错误：在当前目录 {os.getcwd()} 找不到输入文件 {INPUT_FILE}")
        return

    print(f"开始转换数据: {INPUT_FILE} -> {OUTPUT_FILE}")

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    converted_count = 0
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for item in data:
            # 排除非数据字段
            keys = [k for k in item.keys() if k not in ['id', 'fragment_id', 'fragment_hash']]
            if len(keys) < 2: continue

            # 动态识别题目和答案键名
            # 逻辑：最长的字段通常是题目(q)，其次是答案(a)
            sorted_keys = sorted(keys, key=lambda k: len(json.dumps(item[k])), reverse=True)
            q_key, a_key = sorted_keys[0], sorted_keys[1]

            # 转换为 SFT 格式
            instruction_content = extract_text(item[q_key])
            output_content = extract_text(item[a_key])

            sft_item = {
                "instruction": "你是一个自动控制原理专家。请回答以下问题并给出分析：",
                "input": instruction_content,
                "output": output_content
            }

            # 确保每行一个 JSON，且不转义中文/公式
            f_out.write(json.dumps(sft_item, ensure_ascii=False) + "\n")
            converted_count += 1

    print(f"✅ 处理完成！")
    print(f"总计转换条数: {converted_count}")
    print(f"最终产出文件: {os.path.abspath(OUTPUT_FILE)}")


if __name__ == "__main__":
    convert_format()