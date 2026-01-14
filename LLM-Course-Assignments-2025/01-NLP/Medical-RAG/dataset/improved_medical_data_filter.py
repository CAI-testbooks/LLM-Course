import json
import random
import re
from tqdm import tqdm

# ====================== 配置项（可根据需求调整） ======================
# 原始数据路径（JSONL格式）
RAW_DATA_PATH = "/root/autodl-tmp/Medical-RAG/dataset/train_data.json"
# 提取后保存路径
OUTPUT_PATH = "/root/autodl-tmp/Medical-RAG/dataset/medical_qa_8k_high_quality.json"
# 提取数量
EXTRACT_NUM = 8000
# 长度限制（适配Qwen微调）
MAX_QUESTION_LEN = 100 # 增加问题最大字符数，允许更复杂的问题
MIN_QUESTION_LEN = 10   # 问题最小字符数
MAX_ANSWER_LEN = 500   # 增加答案最大字符数，允许更完整回答
MIN_ANSWER_LEN = 50     # 增加答案最小字符数，确保答案有实质内容
# 随机种子（固定，保证结果可复现）
SEED = 42

# ====================== 核心过滤函数 ======================
def is_medical_related(text):
    """过滤纯医疗相关内容 - 增强版"""
    # 医疗领域专业词汇
    medical_keywords = [
        # 基础医疗术语
        "症状", "治疗", "病因", "诊断", "用药", "疾病", "手术", "检查", "疗法", "药物", "处方", "剂量",
        "高血压", "糖尿病", "感冒", "发烧", "咳嗽", "头痛", "胃痛", "心脏病", "脑卒中", "心梗", "肾病",
        # 药物类别
        "抗生素", "抗病毒", "消炎药", "止痛药", "退烧药", "降压药", "降糖药", "激素", "免疫抑制剂",
        # 医疗程序
        "疫苗", "副作用", "禁忌", "注意事项", "不良反应", "适应症", "用法用量", "疗程", "复查", "随访",
        # 检查与指标
        "血常规", "CT", "B超", "MRI", "X光", "心电图", "化验单", "指标", "正常值", "参考范围",
        # 医学专科
        "中医", "西医", "偏方", "理疗", "化疗", "放疗", "药理", "病理", "生理", "解剖",
        # 病理状况
        "感染", "炎症", "肿瘤", "癌症", "遗传", "免疫", "过敏", "疼痛", "水肿", "出血", "梗塞",
        # 身体系统
        "呼吸", "循环", "消化", "神经", "内分泌", "泌尿", "生殖", "皮肤", "骨骼", "肌肉",
        # 专科科室
        "儿科", "妇产科", "眼科", "耳鼻喉科", "口腔科", "骨科", "精神科", "皮肤科", "泌尿科", "心胸外科",
        # 症状描述
        "恶心", "呕吐", "腹泻", "便秘", "头晕", "乏力", "失眠", "食欲", "体重", "发热", "盗汗", "消瘦",
        # 医疗建议
        "饮食", "护理", "康复", "保健", "预防", "调理", "养生", "禁忌", "注意事项", "生活方式",
        # 解剖结构
        "心脏", "肝脏", "肾脏", "肺部", "胃", "肠道", "大脑", "脊髓", "血管", "神经", "关节", "肌肉",
        # 生理指标
        "血压", "血糖", "心率", "体温", "胆固醇", "血脂", "尿酸", "肌酐", "转氨酶", "血红蛋白",
        # 剂量单位
        "mg", "g", "ml", "μg", "毫克", "克", "毫升", "微克", "单位", "IU", "ug", "kg", "cm", "mm",
        # 频率单位
        "每日", "每周", "每月", "一天", "一次", "两次", "三次", "四次", "qd", "bid", "tid", "qid",
        # 时间单位
        "分钟", "小时", "天", "周", "月", "年", "分钟后", "小时后", "天后", "周后", "月后", "年前"
    ]
    
    # 检查医疗专业术语
    text_lower = text.lower()
    medical_term_matches = sum(1 for keyword in medical_keywords if keyword in text_lower)
    
    # 如果包含至少2个医疗术语，则认为是医疗相关
    if medical_term_matches >= 2:
        return True
    
    # 检查是否包含医疗实体
    if has_medical_entities(text):
        return True
    
    # 检查是否包含医疗问句特征
    medical_questions = [
        "怎么治疗", "如何治疗", "吃什么药", "如何用药", "是什么", "怎么办", "原因", "症状", "后果",
        "有什么办法", "如何缓解", "需要注意", "应该注意", "怎么预防", "如何预防", "如何护理"
    ]
    if any(q in text_lower for q in medical_questions):
        return True
    
    return medical_term_matches >= 1

def has_medical_entities(text):
    """检查文本中是否包含医学实体（药物名、疾病名、症状等）"""
    medical_entities = [
        # 常见药物后缀
        "片", "胶囊", "注射液", "口服液", "颗粒", "软膏", "乳膏", "滴眼液", "滴鼻液", "喷雾剂", "贴剂", "药",
        "素", "醇", "酸", "钠", "钙", "镁", "钾", "锌", "铁", "维生素", "胰岛素", "抗生素", "抗病毒",
        # 常见疾病
        "炎", "症", "瘤", "癌", "病", "综合征", "障碍", "缺陷", "损伤", "衰竭", "梗塞", "栓塞", "硬化",
        "增生", "肥大", "萎缩", "坏死", "变性", "化生", "异位", "畸形", "积水", "积气", "积血",
        # 常见症状
        "痛", "疼", "热", "烧", "咳", "喘", "泻", "吐", "晕", "昏", "麻", "瘫", "肿", "痒", "疼", "痒",
        "颤", "抖", "搐", "挛", "挛缩", "僵硬", "乏力", "疲劳", "气促", "心悸", "胸闷", "腹胀",
        # 解剖部位
        "头", "颈", "胸", "腹", "腰", "背", "肩", "臂", "手", "腿", "脚", "眼", "耳", "鼻", "口", "咽"
    ]
    text_lower = text.lower()
    entity_matches = sum(1 for entity in medical_entities if entity in text_lower)
    
    # 如果包含至少2个医学实体，则认为是医疗相关
    return entity_matches >= 2

def check_qa_relevance(question, answer):
    """检查问题和答案的相关性 - 增强版"""
    question_lower = question.lower()
    answer_lower = answer.lower()
    
    # 简单的关键词匹配
    question_keywords = set(re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', question_lower))
    answer_keywords = set(re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', answer_lower))
    
    # 计算关键词重叠率
    if question_keywords:
        overlap_ratio = len(question_keywords.intersection(answer_keywords)) / len(question_keywords)
    else:
        overlap_ratio = 0
    
    # 检查是否包含医学相关实体
    question_has_medical = has_medical_entities(question)
    answer_has_medical = has_medical_entities(answer)
    
    # 检查答案是否直接回答了问题
    # 检查答案中是否包含否定性词语
    negative_words = ["不知道", "不清楚", "不确定", "不详", "未明", "没有", "不能", "不会", "无法", "没有找到", "没有相关"]
    for word in negative_words:
        if word in answer_lower:
            return False, 0  # 返回False和相关性分数
    
    # 检查答案是否过于简短
    if len(answer) < MIN_ANSWER_LEN:
        return False, 0
    
    # 检查答案是否包含通用回复词汇（如"请咨询医生"等）
    generic_responses = ["请咨询医生", "建议就医", "请去医院", "需要面诊", "建议检查", "需要确诊"]
    if any(gr in answer_lower for gr in generic_responses):
        # 这些是合理的回复，不应该过滤掉
        pass
    
    # 检查是否包含"可能"、"也许"、"大概"等不确定词汇
    uncertain_words = ["可能", "也许", "大概", "大概率", "可能情况", "一般认为"]
    uncertain_count = sum(1 for word in uncertain_words if word in answer_lower)
    
    # 综合判断：关键词重叠率 + 医学实体 + 不确定词汇数量
    relevance_score = overlap_ratio * 0.4 + (1 if question_has_medical and answer_has_medical else 0) * 0.4 + (1 if uncertain_count <= 2 else 0) * 0.2
    
    # 设定阈值：相关性分数需大于0.3
    return relevance_score > 0.3, relevance_score

def clean_text(text):
    """清洗文本：去HTML标签、链接、特殊字符、空值"""
    if not isinstance(text, str):
        return ""
    
    # 去除HTML标签
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # 去除URL链接
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
    text = re.sub(r'www\.[^\s]+', ' ', text)
    
    # 去除常见的链接相关字符模式（如您提到的示例）
    text = re.sub(r'href|target|blank|_blank|http|https|www|[a-z]+\.[a-z]{2,}[^\s]*', ' ', text)
    
    # 去除特殊字符（保留中文、数字、常用标点、医学相关的特殊字符和/字符）
    # 保留：中文、数字、英文字母、常见标点、医学单位（mg, g, ml, ug, μg, kg, cm, mm, IU等）
    # 保留：剂量单位、频率单位、时间单位等，以及它们常伴随的/字符
    # 修正转义字符问题
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？：；（）《》、·\-\s\.\%\+\=mggmlugμgkgtblunitIUqd\\bd\\tid\\qid/分钟小时天周月年]', " ", text)
    
    # 去除多余的空格和换行符
    text = re.sub(r'\s+', ' ', text)
    
    # 过滤nan/空字符串
    if text.lower() == "nan" or text.lower() == "null" or len(text) == 0:
        return ""
    
    return text.strip()

def parse_and_filter_item(line):
    """
    解析单条JSONL数据+过滤：
    适配格式：{"questions":[["问题1"],["问题2"]],"answers":["答案"]}
    """
    try:
        item = json.loads(line.strip())
    except:
        return None  # 跳过解析失败的行
    
    # 1. 提取问题和答案（处理二维列表/列表格式）
    # 取第一个问题（多个问题取首个，保证单轮问答）
    questions = item.get("questions", [])
    if not questions or not questions[0]:
        return None
    question = questions[0][0] if isinstance(questions[0], list) else questions[0]
    
    # 取第一个答案
    answers = item.get("answers", [])
    if not answers:
        return None
    answer = answers[0]
    
    # 2. 清洗文本
    clean_q = clean_text(question)
    clean_a = clean_text(answer)
    
    # 如果清洗后内容为空，则跳过
    if not clean_q or not clean_a:
        return None
    
    # 3. 过滤非医疗内容
    if not (is_medical_related(clean_q) or is_medical_related(clean_a)):
        return None
    
    # 4. 长度过滤
    if len(clean_q) < MIN_QUESTION_LEN or len(clean_q) > MAX_QUESTION_LEN:
        return None
    if len(clean_a) < MIN_ANSWER_LEN or len(clean_a) > MAX_ANSWER_LEN:
        return None
    
    # 5. 问答相关性检查 - 现在返回相关性分数
    is_relevant, relevance_score = check_qa_relevance(clean_q, clean_a)
    if not is_relevant:
        return None
    
    return {
        "question": clean_q,
        "answer": clean_a,
        "qa_relevance_score": relevance_score
    }

# ====================== 主提取流程 ======================
def main():
    # 1. 加载并解析JSONL数据（逐行读取）
    print(f"开始加载原始JSONL数据：{RAW_DATA_PATH}")
    filtered_data = []
    with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"原始数据总行数：{len(lines)}")
    
    # 2. 逐行解析+过滤
    print("开始清洗过滤数据...")
    for line in tqdm(lines, desc="过滤进度"):
        parsed_item = parse_and_filter_item(line)
        if parsed_item:
            filtered_data.append(parsed_item)
    
    print(f"基础过滤后数据条数：{len(filtered_data)}")
    
    # 3. 按问答相关性排序，优先保留相关性高的数据
    filtered_data.sort(key=lambda x: x.get('qa_relevance_score', 0), reverse=True)
    
    # 4. 去重（基于问题内容，保留相关性得分最高的版本）
    seen_questions = set()
    unique_data = []
    for item in filtered_data:
        if item["question"] not in seen_questions:
            seen_questions.add(item["question"])
            # 移除评分字段，只保留问题和答案
            unique_data.append({"question": item["question"], "answer": item["answer"]})
    
    print(f"去重后有效数据条数：{len(unique_data)}")
    
    # 5. 随机采样8K条
    if len(unique_data) < EXTRACT_NUM:
        print(f"警告：过滤后仅{len(unique_data)}条有效数据，不足8K，将提取全部")
        final_data = unique_data
    else:
        random.seed(SEED)
        final_data = random.sample(unique_data, EXTRACT_NUM)
    
    # 6. 保存结果（标准JSON数组格式，适配后续微调）
    print(f"开始保存{len(final_data)}条数据到：{OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    
    print(f"高质量数据提取完成！最终数据保存至：{OUTPUT_PATH}")
    print(f"最终提取条数：{len(final_data)}")
    # 打印示例
    print("\n高质量数据示例：")
    for i in range(min(5, len(final_data))):
        print(f"示例{i+1}：")
        print(f"问题：{final_data[i]['question']}")
        print(f"答案：{final_data[i]['answer'][:200]}...")  # 只显示前200字符
        print("-" * 80)

if __name__ == "__main__":
    main()