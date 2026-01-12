#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速RAG测试 - 验证优化效果
"""

import json
import os
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re

@dataclass
class RetrievalConfig:
    """检索配置类"""
    title_weight: float = 0.6
    question_weight: float = 0.4
    answer_weight: float = 0.0
    
    def normalize_weights(self):
        """标准化权重"""
        total = self.title_weight + self.question_weight + self.answer_weight
        if total > 0:
            self.title_weight /= total
            self.question_weight /= total
            self.answer_weight /= total

class QuickRAGSystem:
    """快速RAG系统 - 极致优化版本"""
    
    def __init__(self, max_records: int = 5000):
        """
        初始化快速RAG系统
        """
        self.max_records = max_records
        self.retrieval_config = RetrievalConfig()
        self.retrieval_config.normalize_weights()
        self.data_records = []
        
    def _build_simple_index(self, text: str) -> set:
        """构建简单文本索引"""
        if not text:
            return set()
        words = re.findall(r'\b\w+\b', text.lower())
        return set(words)
    
    def _fast_similarity(self, index1: set, index2: set) -> float:
        """快速相似度计算"""
        if not index1 and not index2:
            return 1.0
        if not index1 or not index2:
            return 0.0
        
        intersection = len(index1.intersection(index2))
        union = len(index1.union(index2))
        return intersection / union if union > 0 else 0.0
    
    def load_and_process_data(self, data_path: str) -> bool:
        """快速数据加载"""
        try:
            print(f"🚀 快速加载数据（限制 {self.max_records} 条）...")
            
            with open(data_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            # 限制记录数量
            raw_data = raw_data[:self.max_records]
            print(f"📥 加载了 {len(raw_data)} 条数据")
            
            start_time = time.time()
            
            for i, item in enumerate(raw_data):
                record = {
                    'index': i,
                    'title': item.get('questionTitle', ''),
                    'question': item.get('questionText', ''),
                    'answer': item.get('answerText', ''),
                    'title_index': self._build_simple_index(item.get('questionTitle', '')),
                    'question_index': self._build_simple_index(item.get('questionText', '')),
                    'answer_index': self._build_simple_index(item.get('answerText', ''))
                }
                self.data_records.append(record)
                
                # 显示进度
                if (i + 1) % 20 == 0:
                    progress = ((i + 1) / len(raw_data)) * 5000
                    elapsed = time.time() - start_time
                    print(f"⚡ 进度: {i+1}/{len(raw_data)} ({progress:.1f}%) - 耗时: {elapsed:.1f}s")
            
            load_time = time.time() - start_time
            print(f"✅ 数据加载完成！处理了 {len(self.data_records)} 条记录，耗时: {load_time:.2f}秒")
            
            return True
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """快速搜索"""
        if not self.data_records:
            return []
        
        print(f"🔍 搜索查询: '{query}'")
        
        start_time = time.time()
        query_index = self._build_simple_index(query)
        
        results = []
        
        for record in self.data_records:
            # 计算各部分相似度
            title_sim = self._fast_similarity(query_index, record['title_index'])
            question_sim = self._fast_similarity(query_index, record['question_index'])
            answer_sim = self._fast_similarity(query_index, record['answer_index'])
            
            # 加权计算最终相似度
            final_sim = (
                self.retrieval_config.title_weight * title_sim +
                self.retrieval_config.question_weight * question_sim +
                self.retrieval_config.answer_weight * answer_sim
            )
            
            if final_sim > 0:
                results.append({
                    'record': record,
                    'similarity': final_sim,
                    'title_sim': title_sim,
                    'question_sim': question_sim,
                    'answer_sim': answer_sim
                })
        
        # 排序并返回前k个
        results.sort(key=lambda x: x['similarity'], reverse=True)
        results = results[:top_k]
        
        search_time = time.time() - start_time
        print(f"⚡ 搜索完成！耗时: {search_time:.3f}秒，找到 {len(results)} 个结果")
        
        return results
    
    def generate_response(self, query: str, top_k: int = 3) -> str:
        """生成回答"""
        results = self.search(query, top_k)
        
        if not results:
            return "没有找到相关结果。"
        
        response = [f"基于 {len(results)} 个相关结果回答 '{query}':\n"]
        
        for i, result in enumerate(results, 1):
            record = result['record']
            sim = result['similarity']
            
            response.append(f"**结果 {i}** (相似度: {sim:.3f})")
            response.append(f"问题: {record['title']}")
            if record['answer']:
                answer_preview = record['answer'][:150] + "..." if len(record['answer']) > 150 else record['answer']
                response.append(f"回答: {answer_preview}")
            response.append("")
        
        return "\n".join(response)


def quick_test():
    """快速测试"""
    print("=" * 60)
    print("⚡ 快速RAG系统性能测试")
    print("=" * 60)
    
    # 初始化系统
    rag = QuickRAGSystem(max_records=5000)  # 只测试5000条记录
    
    data_path = "e:/1__xubin_hu/Program and setting/datasets/Mental_Health_conv/cl_output_file.json"
    
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        return
    
    # 加载数据
    if not rag.load_and_process_data(data_path):
        return
    
    # 测试查询
    test_queries = [
        "panic attack symptoms",
        "anxiety treatment",
        "mental health types"
    ]
    
    print("\n" + "="*50)
    print("🔍 搜索测试")
    print("="*50)
    
    total_search_time = 0
    
    for query in test_queries:
        print(f"\n--- 查询: '{query}' ---")
        start_time = time.time()
        response = rag.generate_response(query, top_k=3)
        search_time = time.time() - start_time
        total_search_time += search_time
        
        print(response)
        print(f"⏱️  本次搜索耗时: {search_time:.3f}秒")
        print("-" * 40)
    
    print(f"\n📊 总测试统计:")
    print(f"  数据记录数: {len(rag.data_records)}")
    print(f"  平均搜索时间: {total_search_time/len(test_queries):.3f}秒")
    print(f"  总搜索时间: {total_search_time:.3f}秒")
    
    print("\n✅ 快速测试完成！")


if __name__ == "__main__":
    quick_test()