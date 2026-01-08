import os
import json
import time
import torch
import jieba
from rouge_chinese import Rouge
from datasets import load_dataset
from tqdm import tqdm
from openai import OpenAI

from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
import chromadb

# ================= ⚙️ 配置区域 =================

# 1. DeepSeek API 配置
DEEPSEEK_API_KEY = "sk-79bbee98a7bf4215a5e27a993f1f0a23"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 2. 待评估的本地模型路径
# MODEL_PATH = "./Qwen/Qwen2.5-7B-Instruct"  # 第一次运行基线模型
MODEL_PATH = "./Qwen/Qwen-Medical-Merged"       # 第二次运行微调后模型

# 3. 其他路径配置
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "medical_rag"
EMBED_PATH = "./BAAI/bge-m3"

# 4. 测试配置
TEST_SIZE = 100  # 测试集样本数量

# ================= 🛠️ 工具函数 =================

def compute_rouge(pred, label):
    # 计算 ROUGE-L 分数 (近似引用 F1)
    if not pred or not label:
        return 0.0
    rouge = Rouge()
    # 中文分词
    pred_seg = ' '.join(jieba.cut(pred))
    label_seg = ' '.join(jieba.cut(label))
    try:
        scores = rouge.get_scores(pred_seg, label_seg)
        return scores[0]['rouge-l']['f']
    except Exception:
        return 0.0

def call_deepseek_judge(prompt, model="deepseek-chat"):
    # 调用 DeepSeek API 进行判别
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个公正、严谨的医疗问答评判专家。请只输出 YES 或 NO，不要包含其他废话。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=10
        )
        return response.choices[0].message.content.strip().upper()
    except Exception as e:
        print(f"⚠️ DeepSeek API 调用失败: {e}")
        return "ERROR"

def evaluate_single_sample(query_engine, question, reference):
    # 评估单条数据：RAG生成 -> Rouge计算 -> DeepSeek打分
    # 1. RAG 生成回答
    response_obj = query_engine.query(question)
    pred_response = response_obj.response
    
    # 获取检索到的上下文
    retrieved_contexts = [n.get_content() for n in response_obj.source_nodes]
    context_text = "\n\n".join(retrieved_contexts)[:3000]

    # 2. 计算 Rouge-L
    rouge_score = compute_rouge(pred_response, reference)

    # 3. DeepSeek 评估准确率
    # 判断生成的回答是否与标准答案意思一致
    acc_prompt = (
        f"【任务】：判断模型回答是否在医学事实层面与标准答案一致。\n"
        f"【问题】：{question}\n"
        f"【标准答案】：{reference}\n"
        f"【模型回答】：{pred_response}\n\n"
        f"请判断：模型回答是否正确？如果是，输出 'YES'；如果错误或答非所问，输出 'NO'。"
    )
    acc_res = call_deepseek_judge(acc_prompt)
    is_accurate = 1 if "YES" in acc_res else 0

    # 4. DeepSeek 评估幻觉
    # 判断回答是否完全基于参考文档 (Faithfulness)
    hal_prompt = (
        f"【任务】：判断模型回答是否包含参考文档中未提及的信息（即幻觉）。\n"
        f"【参考文档】：\n{context_text}\n\n"
        f"【模型回答】：{pred_response}\n\n"
        f"请判断：模型回答中的关键信息是否都能在参考文档中找到支持？\n"
        f"如果包含文档里没说的额外信息（可能是幻觉），输出 'YES'；\n"
        f"如果完全基于文档回答，输出 'NO'。"
    )
    hal_res = call_deepseek_judge(hal_prompt)
    # 这里 YES 代表有幻觉，NO 代表无幻觉
    is_hallucinated = 1 if "YES" in hal_res else 0

    return {
        "question": question,
        "reference": reference,
        "prediction": pred_response,
        "rouge_l": rouge_score,
        "accurate": is_accurate,
        "hallucinated": is_hallucinated,
        "contexts": context_text[:200] + "..."
    }

# ================= 🚀 主程序 =================

def main():
    print(f"🚀 开始评估流程...")
    print(f"📊 当前评估模型: {MODEL_PATH}")
    print(f"⚖️  裁判模型: DeepSeek-V3 (API)")

    # 1. 加载本地 Embedding
    print("Loading Embedding Model...")
    embed_model = HuggingFaceEmbedding(model_name=EMBED_PATH, device="cuda", trust_remote_code=True)
    Settings.embedding_model = embed_model
    
    # 2. 加载本地 LLM (待评估的对象)
    print("Loading Local LLM...")
    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=512,
        tokenizer_name=MODEL_PATH,
        model_name=MODEL_PATH,
        device_map="auto",
        model_kwargs={"torch_dtype": torch.float16, "load_in_4bit": True},
        generate_kwargs={"temperature": 0.1, "do_sample": False}
    )
    Settings.llm = llm

    # 3. 连接向量库
    print("Connecting to ChromaDB...")
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    
    # 构建查询引擎
    query_engine = index.as_query_engine(similarity_top_k=3)

    # 4. 准备测试集
    print("Loading Dataset...")
    dataset = load_dataset("./Huatuo26M-Lite", split="train")
    # 随机打乱并取前 N 个
    test_set = dataset.shuffle(seed=2024).select(range(TEST_SIZE))
    
    results = []
    
    # 5. 循环评估
    print(f"🏁 开始评测 {TEST_SIZE} 条样本 (可能需要几分钟)...")
    for i, item in enumerate(tqdm(test_set)):
        try:
            res = evaluate_single_sample(query_engine, item['question'], item['answer'])
            results.append(res)
            time.sleep(0.5) 
        except Exception as e:
            print(f"❌ 样本 {i} 评估出错: {e}")
            continue

    # 6. 统计结果
    if not results:
        print("❌ 没有生成有效结果。")
        return

    avg_rouge = sum(r['rouge_l'] for r in results) / len(results)
    avg_acc = sum(r['accurate'] for r in results) / len(results)
    avg_hal = sum(r['hallucinated'] for r in results) / len(results)
    
    # 打印最终报告
    print("\n" + "="*40)
    print(f"📝 最终评估报告 - {os.path.basename(MODEL_PATH)}")
    print("="*40)
    print(f"✅ 准确率 (Accuracy):        {avg_acc:.2%}")
    print(f"📚 引用一致性 (Rouge-L):    {avg_rouge:.4f}")
    print(f"👻 幻觉率 (Hallucination):  {avg_hal:.2%}")
    print("="*40)
    
    # 保存 JSON 文件
    output_filename = f"eval_report_{os.path.basename(MODEL_PATH)}.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"💾 详细日志已保存至: {output_filename}")

if __name__ == "__main__":
    main()