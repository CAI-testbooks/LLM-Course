# eval_model.py - 医疗问答系统微调效果评估
import json
import torch
import numpy as np
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from rouge import Rouge
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import re
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd
from langdetect import detect
import warnings
warnings.filterwarnings('ignore')

# 下载必要的NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class MedicalQAEvaluator:
    def __init__(self, base_model_name, fine_tuned_model_path):
        """
        初始化评估器
        :param base_model_name: 基础模型名称
        :param fine_tuned_model_path: 微调后模型路径
        """
        self.base_model_name = base_model_name
        self.fine_tuned_model_path = fine_tuned_model_path
        
        # 初始化基础模型
        self.base_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        if self.base_tokenizer.pad_token is None:
            self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 初始化微调模型
        self.ft_tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path, trust_remote_code=True)
        if self.ft_tokenizer.pad_token is None:
            self.ft_tokenizer.pad_token = self.ft_tokenizer.eos_token
        
        self.ft_model = AutoModelForCausalLM.from_pretrained(
            fine_tuned_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 初始化ROUGE
        self.rouge = Rouge()
        
    def generate_response(self, model, tokenizer, question, max_new_tokens=512):
        """
        生成模型响应
        :param model: 模型
        :param tokenizer: 分词器
        :param question: 问题
        :param max_new_tokens: 最大生成token数
        :return: 生成的响应
        """
        prompt = f"<|system|>\n请回答以下医疗相关问题：\n<|user|>\n{question}\n<|assistant|>\n"
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 提取助手部分的响应
        assistant_part = response.split("<|assistant|>")[-1].strip()
        return assistant_part
    
    def calculate_rouge_scores(self, generated, reference):
        """
        计算ROUGE分数
        :param generated: 生成的文本
        :param reference: 参考文本
        :return: ROUGE分数
        """
        try:
            scores = self.rouge.get_scores(generated, reference)[0]
            return {
                'rouge-1': scores['rouge-1']['f'],
                'rouge-2': scores['rouge-2']['f'],
                'rouge-l': scores['rouge-l']['f']
            }
        except:
            return {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0}
    
    def calculate_bleu_score(self, generated, reference):
        """
        计算BLEU分数
        :param generated: 生成的文本
        :param reference: 参考文本
        :return: BLEU分数
        """
        try:
            generated_tokens = word_tokenize(generated.lower())
            reference_tokens = [word_tokenize(reference.lower())]
            bleu_score = sentence_bleu(reference_tokens, generated_tokens)
            return bleu_score
        except:
            return 0
    
    def detect_hallucination(self, generated, reference):
        """
        检测幻觉 - 基于事实一致性
        :param generated: 生成的文本
        :param reference: 参考文本
        :return: 幻觉率 (0-1)
        """
        # 简化的幻觉检测：检查生成内容与参考内容的语义一致性
        # 这里使用一个简化的检测方法
        gen_lower = generated.lower()
        ref_lower = reference.lower()
        
        # 检查是否包含明显的医疗错误或无关内容
        medical_keywords = [
            '癌症', '肿瘤', '治疗', '药物', '症状', '诊断', '手术', '感染',
            '病毒', '细菌', '抗生素', '免疫', '疫苗', '检查', '化验'
        ]
        
        # 计算生成文本和参考文本的交集
        gen_words = set(gen_lower.split())
        ref_words = set(ref_lower.split())
        
        # 如果生成的内容与参考内容差异很大，可能有幻觉
        intersection = gen_words.intersection(ref_words)
        union = gen_words.union(ref_words)
        
        if len(union) == 0:
            return 1.0  # 完全不一致
        
        # 简化的幻觉检测：基于词汇重叠和医疗关键词的一致性
        overlap_ratio = len(intersection) / len(union) if len(union) > 0 else 0
        
        # 检查生成内容中是否包含医疗关键词但与参考不一致
        gen_medical = any(keyword in gen_lower for keyword in medical_keywords)
        ref_medical = any(keyword in ref_lower for keyword in medical_keywords)
        
        if gen_medical and not ref_medical:
            return 1.0  # 生成了医疗内容但参考没有，可能是幻觉
        elif not gen_medical and ref_medical:
            return 0.8  # 没有生成医疗内容但参考有，可能是遗漏
        else:
            # 都有医疗内容，看一致性
            return 1.0 - overlap_ratio
    
    def calculate_accuracy(self, generated, reference):
        """
        计算准确率 - 基于语义相似度
        :param generated: 生成的文本
        :param reference: 参考文本
        :return: 准确率 (0-1)
        """
        # 使用ROUGE-L作为准确率的近似
        rouge_scores = self.calculate_rouge_scores(generated, reference)
        return rouge_scores['rouge-l']
    
    def evaluate_dataset(self, dataset_path, limit=None):
        """
        评估数据集
        :param dataset_path: 数据集路径
        :param limit: 限制评估样本数
        :return: 评估结果
        """
        # 加载测试数据集
        dataset = load_from_disk(dataset_path)
        test_dataset = dataset["test"] if "test" in dataset else dataset["train"]
        
        if limit:
            test_dataset = test_dataset.select(range(min(limit, len(test_dataset))))
        
        results = {
            'base_model': {
                'accuracy': [],
                'f1_scores': [],
                'rouge_1': [],
                'rouge_2': [],
                'rouge_l': [],
                'bleu': [],
                'hallucination_rate': [],
                'responses': []
            },
            'fine_tuned_model': {
                'accuracy': [],
                'f1_scores': [],
                'rouge_1': [],
                'rouge_2': [],
                'rouge_l': [],
                'bleu': [],
                'hallucination_rate': [],
                'responses': []
            }
        }
        
        for i, example in enumerate(test_dataset):
            print(f"Processing example {i+1}/{len(test_dataset)}")
            
            # 获取问题和参考答案
            questions = example['questions']
            answers = example['answers']
            
            # 处理问题格式
            if isinstance(questions[0], list):
                question = questions[0][0] if len(questions[0]) > 0 else ""
            else:
                question = questions[0]
            
            reference_answer = answers[0] if len(answers) > 0 else ""
            
            # 生成基础模型响应
            base_response = self.generate_response(self.base_model, self.base_tokenizer, question)
            
            # 生成微调模型响应
            ft_response = self.generate_response(self.ft_model, self.ft_tokenizer, question)
            
            # 计算基础模型指标
            base_accuracy = self.calculate_accuracy(base_response, reference_answer)
            base_rouge = self.calculate_rouge_scores(base_response, reference_answer)
            base_bleu = self.calculate_bleu_score(base_response, reference_answer)
            base_hallucination = self.detect_hallucination(base_response, reference_answer)
            
            # 计算微调模型指标
            ft_accuracy = self.calculate_accuracy(ft_response, reference_answer)
            ft_rouge = self.calculate_rouge_scores(ft_response, reference_answer)
            ft_bleu = self.calculate_bleu_score(ft_response, reference_answer)
            ft_hallucination = self.detect_hallucination(ft_response, reference_answer)
            
            # 计算F1分数 (基于BLEU和ROUGE的加权)
            base_f1 = 2 * (base_bleu * base_rouge['rouge-l']) / (base_bleu + base_rouge['rouge-l']) if (base_bleu + base_rouge['rouge-l']) > 0 else 0
            ft_f1 = 2 * (ft_bleu * ft_rouge['rouge-l']) / (ft_bleu + ft_rouge['rouge-l']) if (ft_bleu + ft_rouge['rouge-l']) > 0 else 0
            
            # 保存基础模型结果
            results['base_model']['accuracy'].append(base_accuracy)
            results['base_model']['f1_scores'].append(base_f1)
            results['base_model']['rouge_1'].append(base_rouge['rouge-1'])
            results['base_model']['rouge_2'].append(base_rouge['rouge-2'])
            results['base_model']['rouge_l'].append(base_rouge['rouge-l'])
            results['base_model']['bleu'].append(base_bleu)
            results['base_model']['hallucination_rate'].append(base_hallucination)
            results['base_model']['responses'].append({
                'question': question,
                'reference': reference_answer,
                'generated': base_response
            })
            
            # 保存微调模型结果
            results['fine_tuned_model']['accuracy'].append(ft_accuracy)
            results['fine_tuned_model']['f1_scores'].append(ft_f1)
            results['fine_tuned_model']['rouge_1'].append(ft_rouge['rouge-1'])
            results['fine_tuned_model']['rouge_2'].append(ft_rouge['rouge-2'])
            results['fine_tuned_model']['rouge_l'].append(ft_rouge['rouge-l'])
            results['fine_tuned_model']['bleu'].append(ft_bleu)
            results['fine_tuned_model']['hallucination_rate'].append(ft_hallucination)
            results['fine_tuned_model']['responses'].append({
                'question': question,
                'reference': reference_answer,
                'generated': ft_response
            })
        
        return results

def plot_comparison(results):
    """
    绘制微调前后模型性能对比图
    :param results: 评估结果
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('医疗问答系统微调前后性能对比', fontsize=16)
    
    metrics = [
        ('accuracy', '准确率'),
        ('f1_scores', 'F1分数'),
        ('rouge_1', 'ROUGE-1'),
        ('rouge_2', 'ROUGE-2'),
        ('rouge_l', 'ROUGE-L'),
        ('hallucination_rate', '幻觉率')
    ]
    
    for i, (metric, title) in enumerate(metrics):
        row = i // 3
        col = i % 3
        
        base_values = results['base_model'][metric]
        ft_values = results['fine_tuned_model'][metric]
        
        axes[row, col].hist(base_values, alpha=0.7, label='基础模型', bins=20, density=True)
        axes[row, col].hist(ft_values, alpha=0.7, label='微调模型', bins=20, density=True)
        axes[row, col].set_title(f'{title}分布')
        axes[row, col].set_xlabel(title)
        axes[row, col].set_ylabel('密度')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 创建汇总统计表
    summary_data = []
    for metric, title in metrics[:-1]:  # 不包括幻觉率，因为越低越好
        base_mean = np.mean(results['base_model'][metric])
        ft_mean = np.mean(results['fine_tuned_model'][metric])
        improvement = ft_mean - base_mean
        summary_data.append([title, f"{base_mean:.4f}", f"{ft_mean:.4f}", f"{improvement:.4f}"])
    
    # 幻觉率特殊处理（越低越好）
    hallucination_base = np.mean(results['base_model']['hallucination_rate'])
    hallucination_ft = np.mean(results['fine_tuned_model']['hallucination_rate'])
    hallucination_improvement = hallucination_base - hallucination_ft  # 改进值
    summary_data.append(['幻觉率', f"{hallucination_base:.4f}", f"{hallucination_ft:.4f}", f"{hallucination_improvement:.4f}"])
    
    summary_df = pd.DataFrame(summary_data, columns=['指标', '基础模型', '微调模型', '改进'])
    print("\n=== 性能对比汇总 ===")
    print(summary_df.to_string(index=False))
    
    return fig, summary_df

def main():
    # 配置参数
    BASE_MODEL_NAME = "qwen/Qwen2.5-7B-Instruct"
    FINE_TUNED_MODEL_PATH = "./medical-qwen-finetuned"  # 或者合并后的模型路径
    DATASET_PATH = "/root/autodl-tmp/Medical-RAG/dataset"
    
    # 初始化评估器
    evaluator = MedicalQAEvaluator(BASE_MODEL_NAME, FINE_TUNED_MODEL_PATH)
    
    # 评估数据集（限制样本数以加快评估）
    print("开始评估模型性能...")
    results = evaluator.evaluate_dataset(DATASET_PATH, limit=50)  # 限制为50个样本用于快速评估
    
    # 计算总体统计
    print("\n=== 基础模型性能 ===")
    print(f"准确率: {np.mean(results['base_model']['accuracy']):.4f} ± {np.std(results['base_model']['accuracy']):.4f}")
    print(f"F1分数: {np.mean(results['base_model']['f1_scores']):.4f} ± {np.std(results['base_model']['f1_scores']):.4f}")
    print(f"ROUGE-1: {np.mean(results['base_model']['rouge_1']):.4f} ± {np.std(results['base_model']['rouge_1']):.4f}")
    print(f"ROUGE-2: {np.mean(results['base_model']['rouge_2']):.4f} ± {np.std(results['base_model']['rouge_2']):.4f}")
    print(f"ROUGE-L: {np.mean(results['base_model']['rouge_l']):.4f} ± {np.std(results['base_model']['rouge_l']):.4f}")
    print(f"BLEU: {np.mean(results['base_model']['bleu']):.4f} ± {np.std(results['base_model']['bleu']):.4f}")
    print(f"幻觉率: {np.mean(results['base_model']['hallucination_rate']):.4f} ± {np.std(results['base_model']['hallucination_rate']):.4f}")
    
    print("\n=== 微调模型性能 ===")
    print(f"准确率: {np.mean(results['fine_tuned_model']['accuracy']):.4f} ± {np.std(results['fine_tuned_model']['accuracy']):.4f}")
    print(f"F1分数: {np.mean(results['fine_tuned_model']['f1_scores']):.4f} ± {np.std(results['fine_tuned_model']['f1_scores']):.4f}")
    print(f"ROUGE-1: {np.mean(results['fine_tuned_model']['rouge_1']):.4f} ± {np.std(results['fine_tuned_model']['rouge_1']):.4f}")
    print(f"ROUGE-2: {np.mean(results['fine_tuned_model']['rouge_2']):.4f} ± {np.std(results['fine_tuned_model']['rouge_2']):.4f}")
    print(f"ROUGE-L: {np.mean(results['fine_tuned_model']['rouge_l']):.4f} ± {np.std(results['fine_tuned_model']['rouge_l']):.4f}")
    print(f"BLEU: {np.mean(results['fine_tuned_model']['bleu']):.4f} ± {np.std(results['fine_tuned_model']['bleu']):.4f}")
    print(f"幻觉率: {np.mean(results['fine_tuned_model']['hallucination_rate']):.4f} ± {np.std(results['fine_tuned_model']['hallucination_rate']):.4f}")
    
    # 绘制对比图
    fig, summary_df = plot_comparison(results)
    
    # 保存结果
    with open('evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'base_model': {
                'mean_accuracy': float(np.mean(results['base_model']['accuracy'])),
                'std_accuracy': float(np.std(results['base_model']['accuracy'])),
                'mean_f1': float(np.mean(results['base_model']['f1_scores'])),
                'std_f1': float(np.std(results['base_model']['f1_scores'])),
                'mean_rouge_1': float(np.mean(results['base_model']['rouge_1'])),
                'mean_rouge_2': float(np.mean(results['base_model']['rouge_2'])),
                'mean_rouge_l': float(np.mean(results['base_model']['rouge_l'])),
                'mean_bleu': float(np.mean(results['base_model']['bleu'])),
                'mean_hallucination_rate': float(np.mean(results['base_model']['hallucination_rate']))
            },
            'fine_tuned_model': {
                'mean_accuracy': float(np.mean(results['fine_tuned_model']['accuracy'])),
                'std_accuracy': float(np.std(results['fine_tuned_model']['accuracy'])),
                'mean_f1': float(np.mean(results['fine_tuned_model']['f1_scores'])),
                'std_f1': float(np.std(results['fine_tuned_model']['f1_scores'])),
                'mean_rouge_1': float(np.mean(results['fine_tuned_model']['rouge_1'])),
                'mean_rouge_2': float(np.mean(results['fine_tuned_model']['rouge_2'])),
                'mean_rouge_l': float(np.mean(results['fine_tuned_model']['rouge_l'])),
                'mean_bleu': float(np.mean(results['fine_tuned_model']['bleu'])),
                'mean_hallucination_rate': float(np.mean(results['fine_tuned_model']['hallucination_rate']))
            }
        }, f, ensure_ascii=False, indent=2)
    
    print("\n评估完成！结果已保存到 evaluation_results.json")

if __name__ == "__main__":
    main()