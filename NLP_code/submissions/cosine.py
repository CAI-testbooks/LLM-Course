import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_json_file(file_path):
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None

def extract_text_from_json(json_str):
    """从JSON字符串中提取所有文本内容并拼接"""
    try:
        # 解析JSON字符串
        data = json.loads(json_str)
        
        # 提取answer字段
        answer_text = data.get('answer', '')
        
        # 提取supporting_facts中的所有引用文本
        supporting_texts = []
        supporting_facts = data.get('supporting_facts', [])
        for fact in supporting_facts:
            if isinstance(fact, list) and len(fact) >= 1:
                supporting_texts.append(fact[0])
        
        # 拼接所有文本
        full_text = answer_text + ' ' + ' '.join(supporting_texts)
        return full_text.strip()
    except Exception as e:
        print(f"解析JSON字符串失败: {e}")
        return ""

def calculate_cosine_similarity(text1, text2):
    """计算两个文本的余弦相似度"""
    # 创建TF-IDF向量化器
    vectorizer = TfidfVectorizer(stop_words='english')  # 英文文本使用英文停用词
    
    # 处理空文本情况
    if not text1 and not text2:
        return 1.0  # 两个空文本相似度为1
    elif not text1 or not text2:
        return 0.0  # 一个为空一个不为空相似度为0
    
    # 向量化
    try:
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        # 计算余弦相似度
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity)
    except Exception as e:
        print(f"计算相似度失败: {e}")
        return 0.0

def main():
    # 文件路径
    file_path = "/root/autodl-tmp/test_llama-800_exp3.json"
    
    # 加载数据
    data = load_json_file(file_path)
    if data is None or not isinstance(data, list):
        print("数据加载失败或格式不正确")
        return
    
    # 存储所有相似度结果
    similarity_results = []
    
    # 遍历每个样本
    for idx, sample in enumerate(data):
        #print(f"\n处理样本 ID: {sample.get('id', idx)}")
        
        # 获取generated_output和ground_truth字段
        gen_output = sample.get('generated_output', '{}')
        ground_truth = sample.get('ground_truth', '{}')
        
        # 提取文本
        gen_text = extract_text_from_json(gen_output)
        truth_text = extract_text_from_json(ground_truth)
        
        #print(f"Generated文本: {gen_text}")
        #print(f"Ground Truth文本: {truth_text}")
        
        # 计算相似度
        similarity = calculate_cosine_similarity(gen_text, truth_text)
        similarity_results.append(similarity)
        
        #print(f"余弦相似度: {similarity:.4f}")
    
    # 计算平均相似度
    if similarity_results:
        avg_similarity = np.mean(similarity_results)
        print(f"\n===== 统计结果 =====")
        print(f"样本总数: {len(similarity_results)}")
        print(f"平均余弦相似度: {avg_similarity:.4f}")
        #print(f"相似度列表: {[round(s, 4) for s in similarity_results]}")
    else:
        print("没有计算到任何相似度结果")

if __name__ == "__main__":
    main()