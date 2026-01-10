import json

def add_generated_supporting_facts(input_file, output_file):
    """简化版本的处理函数"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for item in data:
        generated_facts = []
        seen = set()
        
        # 遍历所有statements和citations
        for statement in item.get("model_output", {}).get("statements_with_citations", []):
            for citation in statement.get("citation", []):
                cite_text = citation.get("cite", "").strip()
                sentence_idx = citation.get("start_sentence_idx")
                
                if cite_text and sentence_idx is not None:
                    fact_key = f"{cite_text}|{sentence_idx}"
                    if fact_key not in seen:
                        seen.add(fact_key)
                        generated_facts.append([cite_text, sentence_idx])
        
        # 按句子索引排序
        generated_facts.sort(key=lambda x: x[1])
        item["generated_supporting_facts"] = generated_facts
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成！输出文件: {output_file}")

# 使用示例
input_file = "/root/autodl-tmp/longcite_dev_results_1.4/dev2.0_model_results.json"
output_file = "/root/autodl-tmp/longcite_results_with_generated_facts_1.4.json"

add_generated_supporting_facts(input_file, output_file)