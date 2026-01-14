# src/evaluator.py
import evaluate
import re
from typing import List, Dict


class Evaluator:
    """系统评估器"""

    def __init__(self):
        self.rouge = evaluate.load('rouge')
        self.bleu = evaluate.load('bleu')

    def evaluate_rag(self, predictions: List[str], references: List[str]) -> Dict:
        """评估RAG系统"""
        # ROUGE分数
        rouge_results = self.rouge.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True
        )

        # BLEU分数
        bleu_results = self.bleu.compute(
            predictions=predictions,
            references=[[ref] for ref in references]
        )

        # 幻觉检测（简单版本）
        hallucination_rate = self.detect_hallucinations(
            predictions, references)

        return {
            'rouge': rouge_results,
            'bleu': bleu_results['bleu'],
            'hallucination_rate': hallucination_rate
        }

    def detect_hallucinations(self, predictions: List[str], references: List[str]) -> float:
        """检测幻觉率"""
        hallucination_count = 0
        for pred, ref in zip(predictions, references):
            # 简单的检测：如果预测包含大量不在参考中的实体
            pred_entities = set(re.findall(
                r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', pred))
            ref_entities = set(re.findall(
                r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', ref))

            # 检查是否存在大量未在参考中出现的实体
            novel_entities = pred_entities - ref_entities
            if len(novel_entities) > len(pred_entities) * 0.3:  # 30%的新实体
                hallucination_count += 1

        return hallucination_count / len(predictions) if predictions else 0

    def evaluate_citation(self, predictions: List[str], gold_citations: List[List[str]]) -> Dict:
        """评估引用质量"""
        precisions = []
        recalls = []

        for pred, gold in zip(predictions, gold_citations):
            # 提取预测中的引用
            pred_citations = re.findall(r'\[(\d+)\]', pred)

            if not gold:  # 如果没有金标准引用
                if not pred_citations:
                    precisions.append(1.0)
                    recalls.append(1.0)
                else:
                    precisions.append(0.0)
                    recalls.append(0.0)
            else:
                # 计算精度和召回率
                correct = len(set(pred_citations) & set(gold))
                precision = correct / \
                    len(pred_citations) if pred_citations else 0
                recall = correct / len(gold) if gold else 0

                precisions.append(precision)
                recalls.append(recall)

        avg_precision = sum(precisions) / len(precisions) if precisions else 0
        avg_recall = sum(recalls) / len(recalls) if recalls else 0
        f1 = 2 * avg_precision * avg_recall / \
            (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0

        return {
            'citation_precision': avg_precision,
            'citation_recall': avg_recall,
            'citation_f1': f1
        }
