import os
import json
import time
from rag_system import RAGSystem

# ================= 配置 =================
# 确保你的 .env 文件中有 DEEPSEEK_API_KEY
# 或者在这里临时设置 (不推荐提交到 git)
# os.environ["DEEPSEEK_API_KEY"] = "sk-..."

DATA_PATH = "../data/medical.json"


def ensure_test_data():
    """如果不包含数据文件，创建一个只有5条数据的测试文件"""
    if not os.path.exists("../data"):
        os.makedirs("../data")

    if not os.path.exists(DATA_PATH):
        print(f"⚠️ 未检测到 {DATA_PATH}，正在创建临时测试数据...")
        dummy_data = [
            {
                "instruction": "感冒了嗓子疼怎么办？",
                "output": "感冒嗓子疼建议多喝温水，可以服用蓝芩口服液。饮食要清淡，忌辛辣。"
            },
            {
                "instruction": "糖尿病饮食注意什么？",
                "output": "糖尿病患者应控制糖分摄入，少吃甜食，多吃粗粮和蔬菜。主食要定量。"
            }
        ]
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(dummy_data, f, ensure_ascii=False, indent=2)
        print("✅ 临时测试数据创建完成。")


def main():
    print("=" * 50)
    print("🏥 启动医疗 RAG 系统测试")
    print("=" * 50)

    # 1. 检查数据
    ensure_test_data()

    # 2. 初始化系统
    print("\n[1/4] 正在初始化 RAG 系统 (加载 Embedding 模型)...")
    try:
        rag = RAGSystem()
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return

    # 3. 加载数据并构建索引
    print(f"\n[2/4] 正在加载医疗数据 ({DATA_PATH})...")
    docs = rag.load_medical_data(DATA_PATH)

    if not docs:
        print("❌ 数据加载失败或数据为空")
        return

    print(f"\n[3/4] 正在构建向量数据库 (共 {len(docs)} 条)...")
    # 注意：第一次运行会下载 BGE 模型，可能需要几分钟
    success = rag.build_vectorstore(docs)
    if not success:
        print("❌ 向量库构建失败")
        return

    # 初始化问答链 (连接 DeepSeek)
    rag.init_qa_chain()

    # 4. 进行提问测试
    test_query = "糖尿病平时饮食要注意什么？"
    print(f"\n[4/4] 正在提问: '{test_query}'")
    print("-" * 30)

    # 计时
    start = time.time()
    result = rag.ask_question(test_query)
    duration = time.time() - start

    # 5. 输出结果
    if result["success"]:
        print(f"\n🤖 AI 回答 ({duration:.2f}s):\n")
        print("result!!", result)
        print(result["answer"])

        print("\n📚 引用来源 (RAG 证据):")
        if result["source_documents"]:
            for i, doc in enumerate(result["source_documents"], 1):
                # 打印来源内容的片段
                # content_preview = doc['content'].replace('\n', ' ')[:100]
                content_preview = doc['content'].replace('\n', ' ')
                print(f"  [{i}] {content_preview}...")
        else:
            print("  (无引用来源，可能使用了模型自带知识)")

        print("\n✅ 测试通过！")
    else:
        print(f"\n❌ 提问失败: {result.get('error')}")


if __name__ == "__main__":
    main()
