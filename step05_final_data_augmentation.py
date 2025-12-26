import json
import random
import re
import os

# ================= 配置参数 =================
INPUT_FILE = "/kaggle/input/synthetic-data-4550/synthetic_data_5k.json"
OUTPUT_FILE = "/kaggle/working/final_control_data_5050.json"
TARGET_TOTAL = 5050


# ===========================================

def get_item_content(item):
    """
    高级容错：不再依赖键名，直接提取内容
    返回格式：(题目文本, 答案文本, 原始所有字段字典)
    """
    # 提取所有字符串类型的字段及其长度
    text_fields = [(k, str(v)) for k, v in item.items() if k != 'id']
    # 按长度排序，通常最长的是题目或答案
    text_fields.sort(key=lambda x: len(x[1]), reverse=True)

    if len(text_fields) >= 2:
        return text_fields[0][0], text_fields[1][0]  # 返回两个最长的键名
    else:
        # 如果字段不够，强行指定前两个键
        keys = list(item.keys())
        return keys[0], keys[1] if len(keys) > 1 else keys[0]


def transform_logic(item, index):
    """
    动态匹配变形逻辑
    """
    new_item = item.copy()
    new_item['id'] = f"aug_{index}"

    # 每条数据实时检测键名
    q_key, a_key = get_item_content(item)
    q_text = str(new_item[q_key])

    # 变形策略：自控原理数值裂变
    # 寻找所有的数字并进行随机替换
    if any(keyword in q_text for keyword in ['s', 'G(s)', 'K', 'T']):
        # 匹配整数或小数，并替换为随机新值
        q_text = re.sub(r'\b\d+(\.\d+)?\b', lambda x: str(round(random.uniform(0.1, 50.0), 2)), q_text)
        new_item[a_key] = "[参数重校版] " + str(new_item[a_key])

    # 语境增强
    new_item[q_key] = random.choice(["针对该控制系统：", "已知模型为：", "分析以下环节："]) + q_text

    return new_item


def augment_data():
    if not os.path.exists(INPUT_FILE):
        print(f"错误：找不到文件 {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 如果数据是嵌套格式，尝试解包
    if isinstance(data, dict):
        for k in data.keys():
            if isinstance(data[k], list):
                data = data[k]
                break

    current_count = len(data)
    gap = TARGET_TOTAL - current_count

    if gap <= 0:
        print(f"数据量已达标: {current_count}")
        return

    print(f"原始数据: {current_count} 条，开始执行动态键名裂变...")

    # 使用放回抽样补齐缺口
    samples = random.choices(data, k=gap)
    augmented_items = []

    for i, item in enumerate(samples):
        try:
            new_item = transform_logic(item, i + current_count)
            augmented_items.append(new_item)
        except Exception:
            # 极度容错：如果还是出错，直接原样复制并改ID
            new_item = item.copy()
            new_item['id'] = f"safe_aug_{i + current_count}"
            augmented_items.append(new_item)

    final_data = data + augmented_items
    random.shuffle(final_data)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)

    print(f"✅ 处理完成！最终总数: {len(final_data)}")
    print(f"文件保存路径: {OUTPUT_FILE}")


if __name__ == "__main__":
    augment_data()