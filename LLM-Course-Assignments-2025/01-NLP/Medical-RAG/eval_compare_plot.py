import os
import json
import jieba
import time  # 导入time模块 用于休眠防限流
import numpy as np
from collections import Counter
from openai import OpenAI
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ===================== 可视化依赖 =====================
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (16, 10)

# ===================== 配置区 =====================
os.environ["OPENAI_API_KEY"] = "sk-gyuofotkkugmqvlmcuchjdzmipktruzczqvqtqyiyfqbqvsu"
os.environ["OPENAI_API_BASE"] = "https://api.siliconflow.cn/v1"

# 初始化Judge客户端
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_API_BASE"]
)
JUDGE_MODEL = "Qwen/Qwen2.5-72B-Instruct"

# 文件路径配置
RAG_RESULT_PATH = "/root/autodl-tmp/Medical-RAG/eval_results/rag_top100.json"
BASE_RESULT_PATH = "/root/autodl-tmp/Medical-RAG/eval_results/base_top100.json"
EVAL_OUTPUT_PATH = "/root/autodl-tmp/Medical-RAG/eval_results/final_evaluation.json"
PLOT_SAVE_PATH = "/root/autodl-tmp/Medical-RAG/eval_results/base-rag-evaluation_plots.png"

# 幻觉评估Prompt
HALLUCINATION_PROMPT = """
你是一位严谨的医疗领域专业评估专家，需要判断模型回答是否存在幻觉。
评估规则：
1. 幻觉定义：模型回答中包含与参考答案/权威医疗知识相矛盾的信息，或编造不存在的医疗数据、药物、诊疗方案等。
2. 若参考答案信息不足，以权威医疗共识为判断依据。
3. 仅存在表述冗余、语序差异、同义术语替换不属于幻觉。
4. 明确回答"无法提供确切答案"的情况不计入幻觉。

评估对象：
问题：{instruction}
参考答案：{reference}
模型回答：{answer}

请仅返回以下结果之一，无需任何解释：
- "HALLUCINATION"：存在幻觉
- "NO_HALLUCINATION"：无幻觉
"""

# 医疗F1校准Prompt（原封不动保留）
F1_CALIBRATION_PROMPT = """
你是一个严谨的**医疗领域专业评估专家**，目前需要对医疗问答大模型的输出质量进行评估，评估核心指标为 precision（精确率）、f1（F1 值）。

对于以下评估对象：
问题：{instruction}
参考答案：{reference}
模型回答：{answer}

我已采用词级无序 F1 评估方法，得到基础指标分数：precision={precision},  f1={f1}。
该方法存在一定缺陷（如无法识别医疗术语同义表述、忽略临床逻辑合理性），可能导致分数偏低。请你站在医疗专业角度，结合临床规范与实际诊疗逻辑，判断在原分数基础上可提升的幅度。

请严格遵循以下医疗评估规则：
1.  若参考答案过短、表述模糊或不符合临床指南，可忽略参考答案，直接依据**权威医疗共识**判断模型回答的准确性。
2.  提升后的各项指标值**严禁超过 1.0**，且不得低于原基础分数。
3.  评估需兼顾术语准确性与临床实用性，不可过于严苛：模型回答核心医疗信息正确，仅存在表述冗余或语序差异时，应合理加分。
4.  若原词级 F1 分数已能客观反映模型回答质量，可保持原分数不变。
5.  模型回答与参考答案核心信息一致、语言通顺且符合医疗表述规范，可在原分数基础上适当加分；若存在逻辑自洽的合理延伸（如补充临床用药注意事项），额外酌情加分。
6.  重点比对模型回答与参考答案的**核心医疗关键词**（如疾病名称、药物名称、诊疗方案、剂量单位），关键词匹配度高且无知识性错误时，优先提升 recall 与 f1；无无关信息冗余时，优先提升 precision。
7.  医疗术语存在公认同义表述（如“脑梗死”与“脑梗塞”、“心梗”与“心肌梗死”）时，视为有效匹配，不得因表述差异扣分。
8.  模型回答出现**医疗知识性错误**（如药物适应症混淆、疾病诊断错误、剂量单位错误）时，不得提升分数，维持原基础分。

请直接返回校准后的两个指标值，按 precision、f1 的顺序用英文逗号分隔，无需任何解释或多余内容。
"""

# ===================== 核心评估函数 =====================
def word_level_f1(reference, answer):
    """词级无序F1计算（基于jieba分词） 返回：precision, recall, f1"""
    ref_words = list(jieba.cut(reference.strip()))
    ans_words = list(jieba.cut(answer.strip()))
    
    if not ref_words and not ans_words:
        return 1.0, 1.0, 1.0
    if not ref_words or not ans_words:
        return 0.0, 0.0, 0.0
    
    ref_counter = Counter(ref_words)
    ans_counter = Counter(ans_words)
    
    intersection = 0
    for word in ref_counter:
        if word in ans_counter:
            intersection += min(ref_counter[word], ans_counter[word])
    
    precision = intersection / sum(ans_counter.values()) if sum(ans_counter.values()) > 0 else 0.0
    recall = intersection / sum(ref_counter.values()) if sum(ref_counter.values()) > 0 else 0.0
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return round(precision, 4), round(recall, 4), round(f1, 4)

def call_judge_model(prompt):
    """调用Judge模型获取评估结果【已删除内部sleep，无任何休眠】"""
    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt.strip()}],
            temperature=0.1,
            max_tokens=512,
            timeout=60
        )
        res_content = response.choices[0].message.content.strip()
        return res_content  # 直接返回结果，无sleep
    except Exception as e:
        print(f"\n⚠️ Judge模型调用失败: {e}")
        return None

def calibrate_f1(instruction, reference, answer, base_precision, base_f1):
    """使用LLM校准F1分数"""
    prompt = F1_CALIBRATION_PROMPT.format(
        instruction=instruction, reference=reference, answer=answer,
        precision=base_precision, f1=base_f1
    )
    result = call_judge_model(prompt)
    if not result:
        return base_precision, base_f1
    
    try:
        calibrated_precision, calibrated_f1 = map(float, result.split(","))
        calibrated_precision = max(min(calibrated_precision, 1.0), base_precision)
        calibrated_f1 = max(min(calibrated_f1, 1.0), base_f1)
        return round(calibrated_precision, 4), round(calibrated_f1, 4)
    except:
        return base_precision, base_f1

def evaluate_hallucination(instruction, reference, answer):
    """评估幻觉率"""
    if "根据现有医学资料，我无法提供确切答案，建议咨询专业医生" in answer:
        return "NO_HALLUCINATION"
    
    prompt = HALLUCINATION_PROMPT.format(instruction=instruction, reference=reference, answer=answer)
    result = call_judge_model(prompt)
    if result in ["HALLUCINATION", "NO_HALLUCINATION"]:
        return result
    else:
        return "HALLUCINATION" if "错误" in answer or "不存在" in reference and "存在" in answer else "NO_HALLUCINATION"

def evaluate_model_results(result_path, model_name):
    """评估单个模型的结果（F1、幻觉率）"""
    print(f"\n========== 评估 {model_name} 模型 ==========")
    with open(result_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    total_samples = len(results)
    if total_samples == 0:
        return {}
    
    total_base_precision = 0.0
    total_base_f1 = 0.0
    total_calibrated_precision = 0.0
    total_calibrated_f1 = 0.0
    hallucination_count = 0
    
    for idx, item in enumerate(tqdm(results, desc=f"评估{model_name}")):
        instruction = item["instruction"]
        reference = item["reference"]
        answer = item["answer"]
        
        # 1. 计算基础词级F1
        base_precision, _, base_f1 = word_level_f1(reference, answer)
        # 2. LLM校准F1分数
        cal_precision, cal_f1 = calibrate_f1(instruction, reference, answer, base_precision, base_f1)
        # 3. LLM评估幻觉率
        hallucination_result = evaluate_hallucination(instruction, reference, answer)
        
        # 统计计数
        if hallucination_result == "HALLUCINATION":
            hallucination_count += 1
        
        # 累加分数
        total_base_precision += base_precision
        total_base_f1 += base_f1
        total_calibrated_precision += cal_precision
        total_calibrated_f1 += cal_f1
        
        # 保存单条样本评估结果（不再保存 is_accurate）
        results[idx]["base_metrics"] = {"precision": base_precision, "f1": base_f1}
        results[idx]["calibrated_metrics"] = {"precision": cal_precision, "f1": cal_f1}
        results[idx]["hallucination"] = hallucination_result
        
        # ✅ 每评估完一条样本后休眠10秒，防止API限流
        time.sleep(10)
    
    # 计算整体指标（不含 accuracy）
    overall_metrics = {
        "total_samples": total_samples,
        "base_precision": round(total_base_precision / total_samples, 4),
        "base_f1": round(total_base_f1 / total_samples, 4),
        "calibrated_precision": round(total_calibrated_precision / total_samples, 4),
        "calibrated_f1": round(total_calibrated_f1 / total_samples, 4),
        "hallucination_rate": round(hallucination_count / total_samples, 4)
    }
    
    return {"model_name": model_name, "overall_metrics": overall_metrics, "sample_details": results}

# ===================== 可视化函数 =====================
def plot_evaluation_results(rag_metrics, base_metrics, save_path):
    # All metrics to plot (in order)
    metric_labels = [
        'Base Precision', 'Calibrated Precision',
        'Base F1 Score', 'Calibrated F1 Score',
        'Hallucination Rate'
    ]
    
    # Corresponding scores for RAG and Base models
    rag_scores = [
        rag_metrics['base_precision'],
        rag_metrics['calibrated_precision'],
        rag_metrics['base_f1'],
        rag_metrics['calibrated_f1'],
        rag_metrics['hallucination_rate']
    ]
    
    base_scores = [
        base_metrics['base_precision'],
        base_metrics['calibrated_precision'],
        base_metrics['base_f1'],
        base_metrics['calibrated_f1'],
        base_metrics['hallucination_rate']
    ]
    
    x = np.arange(len(metric_labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Left Y-axis: for F1 / Precision (range 0–1, higher is better)
    bar1 = ax1.bar(x[:-1] - width/2, rag_scores[:-1], width, label='RAG Model', color='#1f77b4', alpha=0.9, edgecolor='black', linewidth=0.8)
    bar2 = ax1.bar(x[:-1] + width/2, base_scores[:-1], width, label='Base Model', color='#ff7f0e', alpha=0.9, edgecolor='black', linewidth=0.8)
    
    ax1.set_ylabel('F1 / Precision (Higher is Better)', fontsize=12, color='black')
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels, fontsize=11)
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)

    # Right Y-axis: for Hallucination Rate (0–1, lower is better)
    ax2 = ax1.twinx()
    bar3 = ax2.bar(x[-1] - width/2, rag_scores[-1], width, label='RAG Hallucination', color='#2ca02c', alpha=0.8, hatch='//', edgecolor='black', linewidth=0.8)
    bar4 = ax2.bar(x[-1] + width/2, base_scores[-1], width, label='Base Hallucination', color='#d62728', alpha=0.8, hatch='\\\\', edgecolor='black', linewidth=0.8)
    
    ax2.set_ylabel('Hallucination Rate (Lower is Better)', fontsize=12, color='gray')
    ax2.set_ylim(0, max(rag_scores[-1], base_scores[-1]) * 1.25 or 0.1)
    ax2.tick_params(axis='y', labelcolor='gray')

    # Unified legend
    bars = [bar1, bar2, bar3, bar4]
    labels = ['RAG (F1/Prec)', 'Base (F1/Prec)', 'RAG Hallucination', 'Base Hallucination']
    ax1.legend(bars, labels, loc='upper left', fontsize=10)

    # Add value labels on top of bars
    def add_value_labels(ax, bars, is_hallucination=False):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + (0.01 if not is_hallucination else 0.005),
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold'
                )

    add_value_labels(ax1, bar1)
    add_value_labels(ax1, bar2)
    add_value_labels(ax2, bar3, is_hallucination=True)
    add_value_labels(ax2, bar4, is_hallucination=True)

    plt.title('Medical Domain: RAG vs Base Model Evaluation Metrics\n(F1, Precision, Hallucination Rate)', fontsize=15, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n✅ Evaluation plot saved to: {save_path}")
# ===================== 主函数 =====================
def main():
    rag_evaluation = evaluate_model_results(RAG_RESULT_PATH, "RAG_Model")
    base_evaluation = evaluate_model_results(BASE_RESULT_PATH, "Base_Model")
    
    final_evaluation = {
        "evaluation_config": {
            "judge_model": JUDGE_MODEL,
            "f1_calculation": "词级无序F1（jieba分词）",
            "calibration_method": "LLM-as-a-Judge（医疗专业校准）",
            "anti_limit_strategy": "✅ 每评估完成1条样本后，统一休眠10秒，防止硅基流动限流"
        },
        "rag_model": rag_evaluation,
        "base_model": base_evaluation,
        "comparison": {
            # accuracy_diff 已移除
            "f1_diff": rag_evaluation["overall_metrics"]["calibrated_f1"] - base_evaluation["overall_metrics"]["calibrated_f1"],
            "hallucination_rate_diff": rag_evaluation["overall_metrics"]["hallucination_rate"] - base_evaluation["overall_metrics"]["hallucination_rate"],
            "precision_diff": rag_evaluation["overall_metrics"]["calibrated_precision"] - base_evaluation["overall_metrics"]["calibrated_precision"]
        }
    }
    
    with open(EVAL_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_evaluation, f, ensure_ascii=False, indent=2)
    
    plot_evaluation_results(rag_evaluation["overall_metrics"], base_evaluation["overall_metrics"], PLOT_SAVE_PATH)

    print("\n" + "="*80)
    print("✅ 最终评估汇总报告 (医疗领域 RAG vs Base)".center(80))
    print("="*80)
    print(f"评估Judge模型: {JUDGE_MODEL}")
    print(f"测试样本总数: {rag_evaluation['overall_metrics']['total_samples']}")
    print(f"评估方法: 词级无序F1 + LLM医疗专业校准 + LLM幻觉判定")
    print(f"防限流策略: ✅ 每评估1条样本后休眠10秒，精准防限流")
    print("="*80)
    
    print("\n【📌 RAG检索增强模型 核心指标】")
    print(f"  ✅ 校准后精确率    : {rag_evaluation['overall_metrics']['calibrated_precision']:.4f}")
    print(f"  ✅ 校准后F1值      : {rag_evaluation['overall_metrics']['calibrated_f1']:.4f}")
    print(f"  ⚠️  幻觉率          : {rag_evaluation['overall_metrics']['hallucination_rate']:.4f}")
    
    print("\n【📌 基础Base模型 核心指标】")
    print(f"  ✅ 校准后精确率    : {base_evaluation['overall_metrics']['calibrated_precision']:.4f}")
    print(f"  ✅ 校准后F1值      : {base_evaluation['overall_metrics']['calibrated_f1']:.4f}")
    print(f"  ⚠️  幻觉率          : {base_evaluation['overall_metrics']['hallucination_rate']:.4f}")
    
    print("\n【📊 模型差异对比 (RAG - Base)】")
    print(f"  📈 F1值提升        : {final_evaluation['comparison']['f1_diff']:+.4f}")
    print(f"  📈 精确率提升      : {final_evaluation['comparison']['precision_diff']:+.4f}")
    print(f"  📉 幻觉率变化      : {final_evaluation['comparison']['hallucination_rate_diff']:+.4f} (负数=降低)")
    print("="*80)
    print(f"\n📋 详细评估结果JSON已保存至: {EVAL_OUTPUT_PATH}")
    print(f"📊 可视化图表已保存至: {PLOT_SAVE_PATH}")

if __name__ == "__main__":
    main()