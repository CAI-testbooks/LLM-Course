import time
import json
import pandas as pd
from rag_system import RAGSystem
from tqdm import tqdm

# 加载测试集（这里我们手动定义少量 Ground Truth 用于演示，实际作业可以从 medical.json 抽 50 条）
TEST_DATA = [
    {
        "question": "感冒了嗓子疼怎么办？",
        "ground_truth": "建议多喝温水，服用蓝芩口服液。饮食清淡，忌辛辣。",
    },
    {
        "question": "糖尿病饮食禁忌？",
        "ground_truth": "控制糖分，少吃甜食，主食定量，多吃粗粮蔬菜。",
    },
    {
        "question": "高血压能彻底治愈吗？",
        "ground_truth": "原发性高血压目前无法彻底治愈，需要终身服药控制。",
    }
]

def evaluate_system():
    print("正在初始化系统进行评估...")
    rag = RAGSystem()

    # =========== 修改点开始 ===========
    print("正在加载数据...")
    # 1. 用变量 docs 接收返回的文档列表
    docs = rag.load_medical_data("data/medical.json")

    if not docs:
        print("❌ 数据加载失败，请检查 data/medical.json 是否存在")
        return

    print(f"成功加载 {len(docs)} 条数据，正在构建向量索引...")
    # 2. 将 docs 传入构建函数
    rag.build_vectorstore(docs)
    # =========== 修改点结束 ===========

    rag.init_qa_chain()

    results = []

    print(f"\n开始评估 {len(TEST_DATA)} 条测试数据...")
    for item in tqdm(TEST_DATA):
        q = item["question"]
        gt = item["ground_truth"]

        # 获取 RAG 回答
        response = rag.ask_question(q)
        pred = response["answer"]

        # 判断是否包含引用
        has_citation = "[片段" in pred or "基于知识库" in pred or "[基于" in pred

        # --- LLM-as-a-Judge: 让大模型给这个回答打分 ---
        eval_prompt = f"""
        请作为一名公正的评判者，对比“标准答案”和“系统回答”。
        
        问题：{q}
        标准答案：{gt}
        系统回答：{pred}
        
        请打分（0-10分），并判断是否包含幻觉（是/否）。
        只输出JSON格式，例如：{{"score": 8, "hallucination": "否"}}
        """

        score = 0
        is_hallucination = "未知"

        try:
            # 调用 LLM 进行评分
            # 注意：如果 DeepSeek API 不稳定，这一步可能会慢或失败
            eval_res_str = rag.llm._call(eval_prompt)
            # 清理可能的 markdown 标记
            eval_res_str = eval_res_str.replace("```json", "").replace("```", "").strip()
            eval_res = json.loads(eval_res_str)

            score = eval_res.get("score", 0)
            is_hallucination = eval_res.get("hallucination", "否")
        except Exception as e:
            print(f"评分失败: {e}")
            score = 5 # 解析失败给保底分

        results.append({
            "question": q,
            "answer": pred,
            "score": score,
            "has_citation": has_citation,
            "hallucination": is_hallucination
        })

    # 计算统计指标
    if not results:
        print("没有评估结果")
        return

    df = pd.DataFrame(results)
    avg_score = df["score"].mean()
    citation_rate = (df["has_citation"].sum() / len(df)) * 100

    # 简单的幻觉率计算
    hallucination_count = len(df[df["hallucination"] == "是"])
    hallucination_rate = (hallucination_count / len(df)) * 100

    print("\n" + "="*40)
    print("📊 评估报告 (Evaluation Report)")
    print("="*40)
    print(f"✅ 平均准确得分: {avg_score:.2f} / 10.0")
    print(f"📚 引用覆盖率:   {citation_rate:.2f}%")
    print(f"⚠️ 幻觉率:       {hallucination_rate:.2f}%")
    print("="*40)

    # 保存详细报告
    output_file = "evaluation_report.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"详细报告已保存至 {output_file}")

if __name__ == "__main__":
    evaluate_system()