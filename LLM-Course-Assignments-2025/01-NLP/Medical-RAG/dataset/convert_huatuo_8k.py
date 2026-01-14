import json
import random

# ====================== 核心配置 ======================
# 1. 原始8K医疗数据路径（注意：是JSON文件，不是.py文件！）
RAW_8K_DATA_PATH = "/root/autodl-tmp/Medical-RAG/dataset/medical_qa_8k_high_quality.json"
# 2. 划分后Alpaca格式文件保存路径
TRAIN_OUTPUT = "/root/autodl-tmp/Medical-RAG/dataset/alpaca_formatted_train_data.json"
VAL_OUTPUT = "/root/autodl-tmp/Medical-RAG/dataset/alpaca_formatted_validation_data.json"
TEST_OUTPUT = "/root/autodl-tmp/Medical-RAG/dataset/alpaca_formatted_test_data.json"
# 3. 随机种子（保证划分结果可复现）
SEED = 42
# 4. 划分比例（18:1:1）
TRAIN_RATIO = 0.9
VAL_RATIO = 0.05
TEST_RATIO = 0.05

# ====================== 数据划分函数 ======================
def split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    将数据按比例划分为train/val/test
    Args:
        data: 原始数据列表（JSON数组）
        train_ratio/val_ratio/test_ratio: 划分比例
        seed: 随机种子
    Returns:
        train_data, val_data, test_data
    """
    # 固定随机种子
    random.seed(seed)
    # 打乱数据（保证划分均匀）
    shuffled_data = random.sample(data, len(data))
    
    # 计算划分索引
    total = len(shuffled_data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    # 划分数据
    train_data = shuffled_data[:train_end]
    val_data = shuffled_data[train_end:val_end]
    test_data = shuffled_data[val_end:]
    
    print(f"数据划分完成：")
    print(f"- 训练集：{len(train_data)} 条")
    print(f"- 验证集：{len(val_data)} 条")
    print(f"- 测试集：{len(test_data)} 条")
    
    return train_data, val_data, test_data

# ====================== Alpaca格式转换函数 ======================
def convert_huatuo_to_alpaca(data, output_file_path):
    """
    将华佗数据（question/answer格式）转换为Alpaca格式（instruction/input/output）
    Args:
        data: 待转换的数据列表（[{"question": "...", "answer": "..."}]）
        output_file_path: 输出文件路径
    """
    alpaca_data = []
    
    for idx, item in enumerate(data, 1):
        try:
            # 适配两种格式：
            # 格式1：提取后的8K数据（question/answer）
            question = item.get("question", "")
            answer = item.get("answer", "")
            
            # 格式2：原始JSONL数据（兼容questions/answers，可选保留）
            if not question:
                questions = item.get("questions", [])
                answers = item.get("answers", [])
                if questions and len(questions[0]) > 0:
                    question = questions[0][0]
                if answers and len(answers) > 0:
                    answer = answers[0]
            
            # 安全检查
            if not question or not answer:
                print(f"警告：第 {idx} 条数据无有效问题/答案，跳过")
                continue
            
            # 转换为Alpaca格式（医疗问答input为空）
            alpaca_entry = {
                "instruction": question,  # 问题作为指令
                "input": "",               # 无额外输入，留空
                "output": answer           # 答案作为输出
            }
            alpaca_data.append(alpaca_entry)
        
        except Exception as e:
            print(f"错误：第 {idx} 条数据转换失败 - {e}，跳过")
            continue
    
    # 保存转换后的数据
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        json.dump(alpaca_data, outfile, ensure_ascii=False, indent=2)
    
    print(f"✅ 转换完成：{len(alpaca_data)} 条有效QA对")
    print(f"📄 保存路径：{output_file_path}\n")

# ====================== 主流程 ======================
if __name__ == "__main__":
    # 1. 加载原始8K医疗数据（JSON数组格式）
    print("===== 1. 加载8K医疗数据 =====")
    try:
        with open(RAW_8K_DATA_PATH, 'r', encoding='utf-8') as f:
            raw_8k_data = json.load(f)
        print(f"成功加载 {len(raw_8k_data)} 条数据\n")
    except Exception as e:
        print(f"加载数据失败：{e}")
        print(f"请检查路径是否正确：{RAW_8K_DATA_PATH}")
        exit(1)
    
    # 2. 按8:1:1划分数据
    print("===== 2. 划分train/val/test =====")
    train_data, val_data, test_data = split_data(
        raw_8k_data,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=SEED
    )
    
    # 3. 转换并保存各数据集
    print("===== 3. 转换为Alpaca格式 =====")
    # 训练集
    convert_huatuo_to_alpaca(train_data, TRAIN_OUTPUT)
    # 验证集
    convert_huatuo_to_alpaca(val_data, VAL_OUTPUT)
    # 测试集
    convert_huatuo_to_alpaca(test_data, TEST_OUTPUT)
    
    print("===== 全部完成 =====")
    print(f"最终文件列表：")
    print(f"- 训练集：{TRAIN_OUTPUT}")
    print(f"- 验证集：{VAL_OUTPUT}")
    print(f"- 测试集：{TEST_OUTPUT}")