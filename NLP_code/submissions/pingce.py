import json
import numpy as np
import sys
import os
import time

# ========== 彻底解决阻塞：禁用所有可能卡的操作 ==========
# 1. 强制指定NLTK路径，彻底禁用自动下载
os.environ['NLTK_DATA'] = '/root/nltk_data'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# 2. 强制stdout无缓冲，日志实时输出（核心解决“看似不动”）
sys.stdout.reconfigure(line_buffering=True)

# ========== 仅加载必要依赖（轻量化） ==========
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# 初始化工具（精准+轻量化）
rouge_scorer_obj = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
smoother = SmoothingFunction().method4  # 精准BLEU平滑

# 评测文件路径
FILE_PATH = "/root/autodl-tmp/test_glm-10K1K_exp2.json"

def main():
    try:
        # ========== 步骤1：读取文件（实时日志） ==========
        print(f"[{time.strftime('%H:%M:%S')}] 步骤1/4：开始读取文件...")
        if not os.path.exists(FILE_PATH):
            raise Exception(f"文件不存在：{FILE_PATH}")
        
        # 小文件直接读取（无需分块）
        with open(FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 兼容entities结构
        if isinstance(data, dict) and "entities" in data:
            text_content = data["entities"][0]["text"]
            data = json.loads(text_content)
        print(f"[{time.strftime('%H:%M:%S')}] 步骤1/4：文件读取完成，原始样本数：{len(data)}")

        # ========== 步骤2：过滤有效样本（精准） ==========
        valid_samples = []
        for s in data:
            if "ground_truth" in s and "generated_output" in s:
                valid_samples.append({
                    "gt": s["ground_truth"],
                    "gen": s["generated_output"]
                })
        total = len(valid_samples)
        print(f"[{time.strftime('%H:%M:%S')}] 步骤2/4：过滤完成，有效样本数：{total}")

        if total == 0:
            raise Exception("无有效样本（需包含ground_truth/generated_output）")

        # ========== 步骤3：计算指标（精准+无逐样本打印） ==========
        print(f"[{time.strftime('%H:%M:%S')}] 步骤3/4：开始计算指标（共{total}样本）...")
        rouge_list = []
        bleu_list = []
        cl_list = []

        # 每200样本打印一次进度（无逐样本输出）
        for idx, sample in enumerate(valid_samples):
            # 精准提取supporting_facts文本
            def extract_text(json_str):
                try:
                    d = json.loads(json_str)
                    return [f[0].strip() for f in d.get("supporting_facts", []) if f[0].strip()]
                except:
                    return []
            
            gt_texts = extract_text(sample["gt"])
            gen_texts = extract_text(sample["gen"])

            # 精准计算ROUGE-L（官方逻辑）
            ref = " ".join(gt_texts)
            cand = " ".join(gen_texts)
            rouge = 0.0
            if ref and cand:
                rouge = rouge_scorer_obj.score(ref, cand)['rougeL'].fmeasure

            # 精准计算BLEU（4权重平均）
            bleu = 0.0
            if ref and cand:
                ref_tok = ref.split()
                cand_tok = cand.split()
                if ref_tok and cand_tok:
                    b1 = sentence_bleu([ref_tok], cand_tok, weights=(1,0,0,0), smoothing_function=smoother)
                    b2 = sentence_bleu([ref_tok], cand_tok, weights=(0.5,0.5,0,0), smoothing_function=smoother)
                    b3 = sentence_bleu([ref_tok], cand_tok, weights=(1/3,1/3,1/3,0), smoothing_function=smoother)
                    b4 = sentence_bleu([ref_tok], cand_tok, weights=(0.25,0.25,0.25,0.25), smoothing_function=smoother)
                    bleu = (b1 + b2 + b3 + b4) / 4

            # 精准计算CL（总字符数）
            cl = sum(len(t) for t in gen_texts)

            # 保存结果
            rouge_list.append(rouge)
            bleu_list.append(bleu)
            cl_list.append(cl)

            # 进度提示（每200样本一次）
            if (idx+1) % 200 == 0 or (idx+1) == total:
                print(f"[{time.strftime('%H:%M:%S')}] 进度：{idx+1}/{total} ({(idx+1)/total*100:.1f}%)")

        # ========== 步骤4：输出精准结果 ==========
        print(f"[{time.strftime('%H:%M:%S')}] 步骤4/4：计算完成，输出结果...")
        avg_rouge = np.mean(rouge_list) if total > 0 else 0.0
        avg_bleu = np.mean(bleu_list) if total > 0 else 0.0
        total_cl = sum(cl_list)
        avg_cl = np.mean(cl_list) if total > 0 else 0.0

        # 仅打印总体结果
        print("\n" + "="*80)
        print("【总体评测结果（精准版）】")
        print(f"样本总数：{total}")
        print(f"平均 ROUGE-L (F1)：{avg_rouge:.6f}")
        print(f"平均 BLEU：{avg_bleu:.6f}")
        print(f"总 Citation Length：{total_cl} 字符")
        print(f"平均 Citation Length：{avg_cl:.2f} 字符/样本")
        print("="*80)

    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] 执行出错：{str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # 直接运行，无多余操作
    main()