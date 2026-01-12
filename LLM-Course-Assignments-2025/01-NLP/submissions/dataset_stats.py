#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计数据集中所有数据的questionTitle、questionText、answerText内容长度的最大值
数据集位置: E:\1__xubin_hu\Program and setting\datasets\Mental_Health_conv\cl_output_file.json
"""

import json
import os
from typing import Dict, List, Any

def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """加载JSON数据文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功加载数据文件: {file_path}")
        print(f"数据集包含 {len(data)} 条记录")
        return data
    except FileNotFoundError:
        print(f"错误: 文件未找到 - {file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"错误: JSON解析失败 - {e}")
        return []
    except Exception as e:
        print(f"错误: 读取文件时发生异常 - {e}")
        return []

def calculate_text_length(text: str) -> int:
    """计算文本长度，处理空值和None值"""
    if text is None:
        return 0
    if isinstance(text, str):
        return len(text)
    return 0

def find_max_lengths(data: List[Dict[str, Any]]) -> tuple[Dict[str, int], Dict[str, Any]]:
    """找出questionTitle、questionText、answerText的最大长度"""
    max_lengths = {
        'questionTitle': 0,
        'questionText': 0,
        'answerText': 0
    }
    
    max_length_records = {
        'questionTitle': None,
        'questionText': None,
        'answerText': None
    }
    
    total_records = len(data)
    processed_records = 0
    
    print("\n开始统计各字段长度...")
    
    for i, record in enumerate(data):
        processed_records += 1
        
        # 显示进度
        if processed_records % 1000 == 0 or processed_records == total_records:
            print(f"已处理 {processed_records}/{total_records} 条记录 ({processed_records/total_records*100:.1f}%)")
        
        # 统计questionTitle长度
        title_length = calculate_text_length(record.get('questionTitle', ''))
        if title_length > max_lengths['questionTitle']:
            max_lengths['questionTitle'] = title_length
            max_length_records['questionTitle'] = record
        
        # 统计questionText长度
        text_length = calculate_text_length(record.get('questionText', ''))
        if text_length > max_lengths['questionText']:
            max_lengths['questionText'] = text_length
            max_length_records['questionText'] = record
        
        # 统计answerText长度
        answer_length = calculate_text_length(record.get('answerText', ''))
        if answer_length > max_lengths['answerText']:
            max_lengths['answerText'] = answer_length
            max_length_records['answerText'] = record
    
    print(f"\n统计完成！共处理了 {processed_records} 条记录")
    
    return max_lengths, max_length_records

def display_results(max_lengths: Dict[str, int], max_length_records: Dict[str, Any]):
    """显示统计结果"""
    print("\n" + "="*80)
    print("数据集文本字段长度统计结果")
    print("="*80)
    
    for field_name, max_length in max_lengths.items():
        print(f"\n{field_name} 最大长度: {max_length}")
        print("-" * 50)
        
        # 显示对应的记录
        record = max_length_records[field_name]
        if record:
            if field_name == 'questionTitle':
                title = record.get('questionTitle', '')
                print(f"对应的问题标题: {title}")
            elif field_name == 'questionText':
                content = record.get('questionText', '')
                if content:
                    display_content = content[:200] + "..." if len(content) > 200 else content
                    print(f"对应的正文内容: {display_content}")
                else:
                    print("对应的正文内容: (为空)")
            elif field_name == 'answerText':
                content = record.get('answerText', '')
                if content:
                    display_content = content[:300] + "..." if len(content) > 300 else content
                    print(f"对应的回答内容: {display_content}")
                else:
                    print("对应的回答内容: (为空)")
    
    print("\n" + "="*80)
    print("汇总信息:")
    print(f"- questionTitle 最大长度: {max_lengths['questionTitle']}")
    print(f"- questionText 最大长度: {max_lengths['questionText']}")
    print(f"- answerText 最大长度: {max_lengths['answerText']}")
    print("="*80)

def main():
    """主函数"""
    print("开始统计Mental Health数据集的文本字段长度...")
    
    # 数据集文件路径
    dataset_path = r"E:\1__xubin_hu\Program and setting\datasets\Mental_Health_conv\cl_output_file.json"
    
    # 检查文件是否存在
    if not os.path.exists(dataset_path):
        print(f"错误: 数据集文件不存在 - {dataset_path}")
        return
    
    # 加载数据
    data = load_json_data(dataset_path)
    if not data:
        print("未能加载数据，程序退出")
        return
    
    # 统计最大长度
    max_lengths, max_length_records = find_max_lengths(data)
    
    # 显示结果
    display_results(max_lengths, max_length_records)

if __name__ == "__main__":
    main()