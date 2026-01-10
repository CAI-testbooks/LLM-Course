import json
import openai
import time
import re
import os

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

def evaluate_supporting_facts(context, query, answer, supporting_facts):
    """
    评估supporting_facts的质量
    """
    # 将supporting_facts转换为可读的字符串
    sf_text = ""
    for i, fact in enumerate(supporting_facts):
        if isinstance(fact, list) and len(fact) > 0:
            sf_text += f"{i+1}. {fact[0]}\n"
    
    # 构建评估问题 - 只要求返回分数
    question = f"""
请对以下支持事实(supporting_facts)进行评分：

上下文(context): {context}
问题(query): {query}
答案(answer): {answer}
支持事实(supporting_facts):
{sf_text}

请从两个方面进行评分（0-10分）：
1. 支持度评分：这些支持事实是否能够充分支持问题的答案？（包括问题中的主语主体是否在支持事实中体现，以让用户可以更好的作证答案，支持事实的充分性）
2. 冗余度评分：这些支持事实是否冗余？（支持事实中是否有过量的与问题无关的内容）

请严格按照以下格式返回评分，不要添加任何其他内容：
支持度评分: X
冗余度评分: Y
"""
    
    response = query_gpt4o_mini(question)
    return response

def parse_scores(response):
    """
    从GPT响应中解析评分
    """
    if not response:
        return None, None
    
    # 使用正则表达式提取评分
    support_match = re.search(r'支持度评分:\s*(\d+)', response)
    redundancy_match = re.search(r'冗余度评分:\s*(\d+)', response)
    
    # 如果第一种格式不匹配，尝试其他可能的格式
    if not support_match:
        support_match = re.search(r'支持度[：:]\s*(\d+)', response)
    if not redundancy_match:
        redundancy_match = re.search(r'冗余度[：:]\s*(\d+)', response)
    
    # 如果还是没匹配到，尝试更宽松的匹配
    if not support_match:
        support_match = re.search(r'支持度\s*(\d+)', response)
    if not redundancy_match:
        redundancy_match = re.search(r'冗余度\s*(\d+)', response)
    
    # 最后尝试直接匹配数字
    if not support_match or not redundancy_match:
        numbers = re.findall(r'\d+', response)
        if len(numbers) >= 2:
            support_score = int(numbers[0])
            redundancy_score = int(numbers[1])
            return support_score, redundancy_score
    
    support_score = int(support_match.group(1)) if support_match else None
    redundancy_score = int(redundancy_match.group(1)) if redundancy_match else None
    
    return support_score, redundancy_score

def process_dataset(input_file, output_file, sample_size=None):
    """
    处理数据集，为每个条目的supporting_facts评分
    """
    # 读取数据集
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"开始处理数据集: {input_file}")
    print(f"总条目数: {len(data)}")
    
    # 如果指定了样本大小，只处理部分数据
    if sample_size and sample_size < len(data):
        data = data[:sample_size]
        print(f"只处理前 {sample_size} 条记录")
    
    # 检查输出文件是否存在，如果存在则加载已处理的数据
    processed_data = []
    start_index = 0
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            processed_data = json.load(f)
            start_index = len(processed_data)
            print(f"检测到已有处理数据，从第 {start_index+1} 条开始继续处理")
    
    # 处理每个条目
    success_count = 0
    error_count = 0
    
    # 打开输出文件用于增量写入
    for i in range(start_index, len(data)):
        item = data[i]
        print(f"处理进度: {i+1}/{len(data)}", end=' ')
        
        # 检查是否包含必要的字段
        if not all(key in item for key in ['context', 'query', 'answer', 'supporting_facts']):
            print(f"- 跳过: 缺少必要字段")
            error_count += 1
            # 仍然添加到处理后的数据中
            processed_data.append(item)
            continue
        
        context = item['context']
        query = item['query']
        answer = item['answer']
        supporting_facts = item['supporting_facts']
        
        # 评估supporting_facts
        evaluation_response = evaluate_supporting_facts(context, query, answer, supporting_facts)
        
        if evaluation_response:
            support_score, redundancy_score = parse_scores(evaluation_response)
            
            # 直接添加顶层字段
            item['support_score'] = support_score
            item['redundancy_score'] = redundancy_score
            
            print(f"- 支持度: {support_score}/10, 冗余度: {redundancy_score}/10")
            success_count += 1
        else:
            print(f"- 评估失败")
            # 如果评估失败，设置分数为None
            item['support_score'] = None
            item['redundancy_score'] = None
            error_count += 1
        
        # 添加到处理后的数据
        processed_data.append(item)
        
        # 每处理一条就保存一次，防止数据丢失
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
        # 添加延迟，避免API限制
        time.sleep(1)
    
    print(f"处理完成!")
    print(f"成功评估: {success_count} 条记录")
    print(f"评估失败: {error_count} 条记录")
    print(f"输出文件: {output_file}")
    
    return output_file

# 主函数
if __name__ == "__main__":
    input_file = "/root/autodl-tmp/ezdata_3_final.json"
    output_file = "/root/autodl-tmp/ezdata_3_with_score.json"
    
    # 直接处理全部数据
    print("开始处理数据集...")
    process_dataset(input_file, output_file)