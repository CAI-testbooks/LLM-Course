import json
import openai
import time
import re
import os
from datetime import datetime

def query_gpt4o_mini(question):
    """
    使用GPT-4o-mini模型进行查询
    """
    openai.api_key = "sk-tV4cZ8IDjmMTz3DgjKQKQHa1WP35TM2HhD0Dpdw0pC2m1Ko7"
    openai.base_url = 'https://4.0.wokaai.com/v1/'

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",  # 使用GPT-4o-mini模型
            messages=[
                {"role": "system", "content": "你是一个专业的文本评估专家，擅长评估支持事实的质量。请只返回分数，不需要任何解释。"},
                {"role": "user", "content": question}
            ],
            temperature=0.1  # 降低随机性，确保评分一致性
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API调用错误: {e}")
        return None

def parse_input_content(input_str):
    """
    从input字段中拆分Context和Question
    输入格式示例：Context: xxx Question: xxx
    """
    # 拆分Context和Question（兼容换行/空格等格式）
    context_match = re.search(r'Context:\s*(.*?)\s*Question:', input_str, re.DOTALL)
    question_match = re.search(r'Question:\s*(.*)', input_str, re.DOTALL)
    
    context = context_match.group(1).strip() if context_match else ""
    question = question_match.group(1).strip() if question_match else ""
    
    return context, question

def evaluate_supporting_facts(context, query, ground_truth_answer, generated_supporting_facts):
    """
    评估generated_supporting_facts的质量（0-100分）
    """
    # 将supporting_facts转换为可读的字符串
    sf_text = ""
    for i, fact in enumerate(generated_supporting_facts):
        if isinstance(fact, list) and len(fact) > 0:
            sf_text += f"{i+1}. {fact[0]}\n"
    
    # 构建评估问题 - 调整为0-100分，且冗余度分数越高=越不冗余
    question = f"""
请对以下生成的支持事实(supporting_facts)进行评分（评分范围0-100分）：

上下文(context): {context}
问题(query): {query}
标准答案(ground_truth_answer): {ground_truth_answer}
生成的支持事实(generated_supporting_facts):
{sf_text}

请从两个方面进行评分（0-100分）：
1. 支持度评分：分数越高，表示这些支持事实越能充分支持问题的标准答案；分数越低，表示支持性越差。
   评分维度包括：问题中的核心主体是否在支持事实中体现、支持事实是否能有效佐证答案、信息是否完整。
2. 冗余度评分：分数越高，表示这些支持事实越简洁、无冗余；分数越低，表示支持事实中包含越多与问题无关的冗余内容。

请严格按照以下格式返回评分，不要添加任何其他内容：
支持度评分: X
冗余度评分: Y
"""
    
    response = query_gpt4o_mini(question)
    return response

def parse_scores(response):
    """
    从GPT响应中解析0-100分的评分
    """
    if not response:
        return None, None
    
    # 使用正则表达式提取评分（适配中文冒号/英文冒号、空格）
    support_match = re.search(r'支持度评分:\s*(\d{1,3})', response)
    redundancy_match = re.search(r'冗余度评分:\s*(\d{1,3})', response)
    
    # 兼容其他格式
    if not support_match:
        support_match = re.search(r'支持度[：:]\s*(\d{1,3})', response)
    if not redundancy_match:
        redundancy_match = re.search(r'冗余度[：:]\s*(\d{1,3})', response)
    
    # 最后尝试提取前两个数字
    if not support_match or not redundancy_match:
        numbers = re.findall(r'\d+', response)
        if len(numbers) >= 2:
            # 确保分数在0-100范围内
            support_score = min(max(int(numbers[0]), 0), 100)
            redundancy_score = min(max(int(numbers[1]), 0), 100)
            return support_score, redundancy_score
    
    # 提取并校验分数范围
    support_score = int(support_match.group(1)) if support_match else None
    redundancy_score = int(redundancy_match.group(1)) if redundancy_match else None
    
    # 确保分数在0-100之间
    if support_score is not None:
        support_score = min(max(support_score, 0), 100)
    if redundancy_score is not None:
        redundancy_score = min(max(redundancy_score, 0), 100)
    
    return support_score, redundancy_score

def process_single_dataset(input_file, output_file, sample_size=None):
    """
    处理单个数据集，返回处理结果（成功数/失败数）
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        return 0, 0, 0  # 总条数、成功数、失败数
    
    # 读取数据集
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_count = len(data)
    print(f"\n📂 开始处理数据集: {input_file}")
    print(f"📊 总条目数: {total_count}")
    
    # 如果指定了样本大小，只处理部分数据
    if sample_size and sample_size < total_count:
        data = data[:sample_size]
        print(f"🔍 只处理前 {sample_size} 条记录")
        total_count = sample_size
    
    # 检查输出文件是否存在，如果存在则加载已处理的数据
    processed_data = []
    start_index = 0
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            processed_data = json.load(f)
            start_index = len(processed_data)
            print(f"🔄 检测到已有处理数据，从第 {start_index+1} 条开始继续处理")
    
    # 处理每个条目
    success_count = 0
    error_count = 0
    
    for i in range(start_index, len(data)):
        item = data[i]
        print(f"⏳ 处理进度: {i+1}/{len(data)}", end=' ')
        
        # 检查是否包含必要的字段
        required_fields = ['instruction', 'input', 'ground_truth', 'generated_output']
        if not all(key in item for key in required_fields):
            missing_fields = [k for k in required_fields if k not in item]
            print(f"- ❌ 跳过: 缺少字段 {missing_fields}")
            error_count += 1
            processed_data.append(item)
            continue
        
        try:
            # 1. 解析input中的Context和Question
            context, query = parse_input_content(item['input'])
            if not context or not query:
                print(f"- ❌ 跳过: 无法解析Context/Question")
                error_count += 1
                processed_data.append(item)
                continue
            
            # 2. 解析ground_truth（JSON字符串）提取标准答案
            ground_truth = json.loads(item['ground_truth'])
            ground_truth_answer = ground_truth.get('answer', '')
            
            # 3. 解析generated_output（JSON字符串）提取需要评分的supporting_facts
            generated_output = json.loads(item['generated_output'])
            generated_sf = generated_output.get('supporting_facts', [])
            
            # 4. 评估supporting_facts
            evaluation_response = evaluate_supporting_facts(context, query, ground_truth_answer, generated_sf)
            
            if evaluation_response:
                support_score, redundancy_score = parse_scores(evaluation_response)
                
                # 添加评分字段到原数据
                item['support_score'] = support_score
                item['redundancy_score'] = redundancy_score
                
                print(f"- ✅ 支持度: {support_score}/100, 冗余度: {redundancy_score}/100")
                success_count += 1
            else:
                print(f"- ❌ 评估失败: API返回空")
                item['support_score'] = None
                item['redundancy_score'] = None
                error_count += 1
        
        except json.JSONDecodeError as e:
            print(f"- ❌ 评估失败: JSON解析错误: {str(e)[:50]}...")
            item['support_score'] = None
            item['redundancy_score'] = None
            error_count += 1
        except Exception as e:
            print(f"- ❌ 评估失败: 未知错误: {str(e)[:50]}...")
            item['support_score'] = None
            item['redundancy_score'] = None
            error_count += 1
        
        # 添加到处理后的数据并保存（增量写入）
        processed_data.append(item)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
        # 添加延迟，避免API限制
        time.sleep(1)
    
    print(f"✅ 数据集 {input_file} 处理完成!")
    print(f"   成功评估: {success_count} 条 | 评估失败: {error_count} 条")
    return total_count, success_count, error_count

def main():
    """
    主函数：批量处理多个数据集
    你只需修改下面的 DATASET_LIST 列表即可！
    """
    # ====================== 数据集配置列表（核心修改区）======================
    # 格式说明：
    # {
    #   "input_file": 输入数据集路径,
    #   "output_file": 输出带评分的数据集路径,
    #   "sample_size": 可选，测试时用，比如只处理前10条，正式运行设为None
    # }
    DATASET_LIST = [
        # 示例1：第一个数据集
        #{
            #"input_file": "/root/autodl-tmp/test_Qwen2.5-7B-Instruct_exp3_evaluation.json",
            #"output_file": "/root/autodl-tmp/test_qwen_with_score.json",
            #"sample_size": None
        #},
        {
            "input_file": "/root/autodl-tmp/test_qwen-0.5K_exp3.json",
            "output_file": "/root/autodl-tmp/test_qwen_0.5K_with_score.json",
            "sample_size": None
        },
        {
            "input_file": "/root/autodl-tmp/test_qwen-1K_exp3.json",
            "output_file": "/root/autodl-tmp/test_qwen_1K_with_score.json",
            "sample_size": None
        },
        {
            "input_file": "/root/autodl-tmp/test_qwen-10K_exp3.json",
            "output_file": "/root/autodl-tmp/test_qwen_10K_with_score.json",
            "sample_size": None
        },
        {
            "input_file": "/root/autodl-tmp/test_qwen-10K0.5K_exp3.json",
            "output_file": "/root/autodl-tmp/test_qwen_10K0.5K_with_score.json",
            "sample_size": None
        },
        {
            "input_file": "/root/autodl-tmp/test_qwen-10K1K_exp3.json",
            "output_file": "/root/autodl-tmp/test_qwen_10K1K_with_score.json",
            "sample_size": None
        },
        {
            "input_file": "/root/autodl-tmp/test_glm_exp2.json",
            "output_file": "/root/autodl-tmp/test_glm_with_score.json",
            "sample_size": None
        },
        {
            "input_file": "/root/autodl-tmp/test_glm-0.5K_exp2.json",
            "output_file": "/root/autodl-tmp/test_glm-0.5K_with_score.json",
            "sample_size": None
        },
        {
            "input_file": "/root/autodl-tmp/test_glm-1K_exp2.json",
            "output_file": "/root/autodl-tmp/test_glm-1K_with_score.json",
            "sample_size": None
        },
        {
            "input_file": "/root/autodl-tmp/test_glm-10K_exp2.json",
            "output_file": "/root/autodl-tmp/test_glm-10K_with_score.json",
            "sample_size": None
        },
        {
            "input_file": "/root/autodl-tmp/test_glm-10K0.5K_exp2.json",
            "output_file": "/root/autodl-tmp/test_glm-10K0.5K_with_score.json",
            "sample_size": None
        },
        {
            "input_file": "/root/autodl-tmp/test_glm-10K1K_exp2.json",
            "output_file": "/root/autodl-tmp/test_glm-10K1K_with_score.json",
            "sample_size": None
        },
        
        
    ]
    # ====================== 配置结束 =======================

    # 批量处理汇总统计
    summary = {
        "total_datasets": len(DATASET_LIST),
        "processed_datasets": 0,
        "total_records": 0,
        "success_records": 0,
        "failed_records": 0,
        "failed_datasets": []
    }

    print("="*80)
    print(f"🚀 开始批量处理数据集 | 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📋 待处理数据集总数: {summary['total_datasets']}")
    print("="*80)

    # 遍历处理每个数据集
    for idx, dataset in enumerate(DATASET_LIST, 1):
        print(f"\n{'='*20} 处理第 {idx}/{summary['total_datasets']} 个数据集 {'='*20}")
        input_file = dataset["input_file"]
        output_file = dataset["output_file"]
        sample_size = dataset.get("sample_size", None)

        # 处理单个数据集
        total, success, failed = process_single_dataset(input_file, output_file, sample_size)
        
        # 更新汇总统计
        summary["total_records"] += total
        summary["success_records"] += success
        summary["failed_records"] += failed
        summary["processed_datasets"] += 1

        if failed == total and total > 0:  # 该数据集全部失败
            summary["failed_datasets"].append(input_file)

    # 输出最终汇总报告
    print("\n" + "="*80)
    print("📊 批量处理汇总报告")
    print("="*80)
    print(f"总数据集数: {summary['total_datasets']}")
    print(f"已处理数据集数: {summary['processed_datasets']}")
    print(f"总记录数: {summary['total_records']}")
    print(f"成功评分记录数: {summary['success_records']}")
    print(f"失败评分记录数: {summary['failed_records']}")
    if summary["total_records"] > 0:
        success_rate = (summary["success_records"] / summary["total_records"]) * 100
        print(f"整体成功率: {success_rate:.2f}%")
    if summary["failed_datasets"]:
        print(f"完全处理失败的数据集: {summary['failed_datasets']}")
    print(f"🕒 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == "__main__":
    main()