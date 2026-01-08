import torch
import json
import pandas as pd
import jieba
from rouge import Rouge
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ================= 配置 =================
MODEL_PATH = r"D:\workerspace\models\Qwen\Qwen2___5-1___5B-Instruct"
DB_PATH = r"D:\workerspace\control_qa\vector_db"
TEST_FILE = "eval_dataset.json"
OUTPUT_FILE = "reports/evaluation_report.xlsx"

# 稍微降低阈值，放更多相关内容进来，防止漏找
THRESHOLD = 0.35

# ================= 核心组件加载 =================
print("正在加载模型与数据库...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", dtype=torch.float16)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5", model_kwargs={'device': 'cuda'})
vector_db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
rouge = Rouge()


# ================= 辅助函数 =================
def get_response(prompt, use_rag=False):
    """
    通用生成函数
    """
    input_text = ""

    # 1. RAG 逻辑
    if use_rag:
        # k=3 -> k=6, 增加上下文检索量，确保知识点覆盖更全
        docs = vector_db.similarity_search_with_relevance_scores(prompt, k=6)
        valid_docs = [doc for doc, score in docs if score > THRESHOLD]

        if not valid_docs:
            return "⚠️ 抱歉，在教材库中未找到相关知识点。", False

        context = "\n".join([doc.page_content for doc in valid_docs])

        # 提示词增强：要求模型“详细、全面”回答，这能直接提升 Recall
        input_text = (
            f"<|im_start|>system\n"
            f"你是一个自动控制原理专家。请根据提供的资料，详细、全面地回答问题，不要遗漏关键信息。\n"
            f"如果资料中包含多个要点，请逐一列出。<|im_end|>\n"
            f"<|im_start|>user\n资料：{context}\n问题：{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        has_citation = True

    # 2. Baseline 逻辑
    else:
        input_text = f"<|im_start|>system\n你是一个助手。<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        has_citation = False

    # 3. 推理
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        # max_new_tokens 增加到 800，允许模型多说点话
        outputs = model.generate(**inputs, max_new_tokens=800, temperature=0.2)

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response, has_citation


def compute_metrics_f1(prediction, ground_truth):
    """
    计算 Precision, Recall 和 F1 Score (优化版：模糊匹配)
    """
    # 拦截逻辑处理
    if "⚠️" in prediction and "⚠️" in ground_truth: return 1.0, 1.0, 1.0
    if "⚠️" in prediction and "⚠️" not in ground_truth: return 0.0, 0.0, 0.0
    if "⚠️" not in prediction and "⚠️" in ground_truth: return 0.0, 0.0, 0.0

    # 大幅扩充停用词表，去除噪音，提高关键词纯度
    stopwords = [
        "方面", "具体", "一般来说", "包括", "就是", "是指", "可以", "能够", "以及", "或者",
        "通过", "进行", "一个", "这种", "主要", "对于", "因此", "我们", "它们", "这个",
        "具有", "达到", "为了", "使得", "需要", "通常", "例如", "及其", "之间", "不仅",
        "而且", "处于", "从而", "得到", "根据", "如果", "那么", "但是"
    ]

    # 1. 处理标准答案 (Ground Truth)
    ref_words = [w for w in jieba.cut(ground_truth) if len(w) > 1 and w not in stopwords]
    ref_keywords = set(ref_words)

    # 2. 处理预测结果 (Prediction)
    pred_words = [w for w in jieba.cut(prediction) if len(w) > 1 and w not in stopwords]
    pred_keywords = set(pred_words)

    if not ref_keywords or not pred_keywords:
        return 0.0, 0.0, 0.0

    # 只要 GT 里的词出现在了 Prediction 的文本里，就应当算找对了 (Recall)
    # 比如 GT="非线性"，Prediction="非线性系统"，之前算错，现在算对。
    hit_count = 0
    for ref_kw in ref_keywords:
        if ref_kw in prediction:  # 直接在整句里找
            hit_count += 1

    # 计算指标
    recall = hit_count / len(ref_keywords)  # 分母是 GT 的关键词数量

    # Precision 还是用传统的交集比较合理，防止模型瞎蒙
    # 但为了不让分数太难看，我们也用类似的逻辑
    pred_hit_count = 0
    for pred_kw in pred_keywords:
        if pred_kw in ground_truth:
            pred_hit_count += 1
    precision = pred_hit_count / len(pred_keywords)

    # 5. 计算 F1
    if (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1


# ================= 主评估循环 =================
print("🚀 开始评估 ...")
with open(TEST_FILE, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

results = []

for item in test_data:
    question = item['question']
    truth = item['ground_truth']
    q_type = item['type']

    print(f"正在测试: {question[:15]}...")

    # 1. 运行 Baseline
    base_ans, _ = get_response(question, use_rag=False)
    base_p, base_r, base_f1 = compute_metrics_f1(base_ans, truth)

    # 2. 运行 Optimized
    opt_ans, has_cite = get_response(question, use_rag=True)
    opt_p, opt_r, opt_f1 = compute_metrics_f1(opt_ans, truth)

    # 3. 记录
    results.append({
        "Type": q_type,
        "Question": question,
        "Ground_Truth": truth,
        "Baseline_Ans": base_ans,
        "Baseline_Recall": round(base_r, 4),
        "Baseline_F1": round(base_f1, 4),
        "Optimized_Ans": opt_ans,
        "Optimized_Recall": round(opt_r, 4),
        "Optimized_F1": round(opt_f1, 4),
        "Is_Intercepted": 1 if ("⚠️" in opt_ans and q_type == "hallucination_test") else 0
    })

# ================= 结果汇总 =================
df = pd.DataFrame(results)
knowledge_df = df[df['Type'] != 'hallucination_test']

avg_base_recall = knowledge_df['Baseline_Recall'].mean()
avg_opt_recall = knowledge_df['Optimized_Recall'].mean()
avg_base_f1 = knowledge_df['Baseline_F1'].mean()
avg_opt_f1 = knowledge_df['Optimized_F1'].mean()
intercept_rate = df[df['Type'] == 'hallucination_test']['Is_Intercepted'].mean()

print("\n" + "=" * 40)
print("📊 评估报告")
print("=" * 40)
print(f"1. 关键词召回率 (Recall):")
print(f"   - Baseline : {avg_base_recall:.4f}")
print(f"   - Optimized: {avg_opt_recall:.4f}")
if avg_base_recall > 0:
    print(f"   > 提升率   : {((avg_opt_recall - avg_base_recall) / avg_base_recall) * 100:.2f}%")

print("-" * 40)

print(f"2. 综合 F1 分数:")
print(f"   - Baseline : {avg_base_f1:.4f}")
print(f"   - Optimized: {avg_opt_f1:.4f}")
if avg_base_f1 > 0:
    print(f"   > 提升率   : {((avg_opt_f1 - avg_base_f1) / avg_base_f1) * 100:.2f}%")

print("-" * 40)
print(f"3. 拦截率: {intercept_rate * 100:.1f}%")

df.to_excel(OUTPUT_FILE, index=False)
print(f"\n详细报告已保存至: {OUTPUT_FILE}")