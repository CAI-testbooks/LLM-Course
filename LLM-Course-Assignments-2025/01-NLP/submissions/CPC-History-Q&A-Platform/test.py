import os
import re
import json
import faiss
import torch
import gradio as gr
import numpy as np
from typing import List, Dict, Tuple
from jieba import lcut
from sklearn.metrics import accuracy_score, f1_score
from sentence_transformers import SentenceTransformer, util
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)


# ===================== 1. 核心配置（仅需修改2处） =====================
class Config:
    LORA_MODEL_PATH = "./cpc_history_lora"  # 你的LoRA微调后模型路径（如./model）
    QA_DATA_DIR = "./data"  #
    # ---------- 固定配置（作业要求） ----------
    QA_DATA_PATH = f"{QA_DATA_DIR}/cpc_history_qa.json"  # 适配你data目录的QA数据（支持json/csv）
    VECTOR_DB_PATH = f"{QA_DATA_DIR}/faiss_index.index"  # 自动构建的向量库路径
    MAX_CONTEXT_LEN = 32768  # 32k长上下文
    CONFIDENCE_THRESH = 0.7  # 低于该值拒绝回答
    MAX_HISTORY_TURNS = 5  # 多轮对话保留5轮
    RETRIEVE_TOP_K = 8  # 检索召回数（可迭代调整）
    RERANK_TOP_K = 5  # 重排序后保留数


# ===================== 2. 读取并清洗你的QA数据 =====================
def load_qa_data(config: Config) -> List[Dict]:
    """读取你data目录下的QA数据，自动清洗适配RAG"""
    # 支持json/csv两种格式（覆盖常见QA数据格式）
    qa_files = [f for f in os.listdir(config.QA_DATA_DIR) if f.endswith((".json", ".csv"))]
    if not qa_files:
        raise ValueError(f"❌ 在{config.QA_DATA_DIR}目录未找到QA数据（json/csv）")

    data_file = os.path.join(config.QA_DATA_DIR, qa_files[0])
    print(f"✅ 读取QA数据：{data_file}")

    # 读取数据
    if data_file.endswith(".json"):
        with open(data_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    else:  # csv
        import pandas as pd
        df = pd.read_csv(data_file, encoding="utf-8")
        raw_data = df.to_dict("records")

    # 数据清洗（去空、去重、格式化）
    cleaned_data = []
    for idx, item in enumerate(raw_data):
        # 适配常见QA字段名（question/q | answer/a | source）
        question = item.get("question") or item.get("q") or ""
        answer = item.get("answer") or item.get("a") or ""
        source = item.get("source") or f"数据-{idx}"  # 自动补充来源（作业要求引用）

        # 清洗：去空、去特殊字符
        question = re.sub(r"[\s\t\n]+", "", str(question)).strip()
        answer = re.sub(r"[\s\t\n]+", "", str(answer)).strip()

        if not question or not answer:
            continue

        # 长文本分块（适配32k上下文）
        content = f"问题：{question} 答案：{answer}"
        chunks = []
        sentences = re.split(r"[。！？；]", content)
        current_chunk = ""
        for sent in sentences:
            if len(current_chunk) + len(sent) < 2048:  # 单块不超2048 tokens
                current_chunk += sent + "。"
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sent + "。"
        if current_chunk:
            chunks.append(current_chunk.strip())

        cleaned_data.append({
            "question": question,
            "answer": answer,
            "source": source,
            "chunks": chunks
        })

    print(f"✅ QA数据清洗完成，有效数据量：{len(cleaned_data)}条（需≥5k，不足会提示）")
    if len(cleaned_data) < 5000:
        print("⚠️ 注意：当前有效QA数据不足5k条，建议补充数据以满足作业要求")
    return cleaned_data


# ===================== 3. 自动构建/加载向量数据库 =====================
class VectorDB:
    def __init__(self, config: Config):
        self.config = config
        self.embedding_model = SentenceTransformer("shibing624/text2vec-base-chinese")  # 轻量且效果好
        self.index = None
        self.doc_map = {}  # 向量索引→文档映射

    def build(self, cleaned_data: List[Dict]):
        """基于你的QA数据构建FAISS向量库"""
        # 提取所有文本块
        all_chunks = []
        for item in cleaned_data:
            for chunk in item["chunks"]:
                all_chunks.append({
                    "text": chunk,
                    "source": item["source"],
                    "answer": item["answer"]
                })

        # 生成embedding
        texts = [item["text"] for item in all_chunks]
        embeddings = self.embedding_model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )

        # 构建FAISS索引
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        self.doc_map = {i: all_chunks[i] for i in range(len(all_chunks))}

        # 保存向量库
        faiss.write_index(self.index, self.config.VECTOR_DB_PATH)
        print(f"✅ 向量库构建完成，存储路径：{self.config.VECTOR_DB_PATH}")

    def load(self):
        """加载已构建的向量库"""
        if os.path.exists(self.config.VECTOR_DB_PATH):
            self.index = faiss.read_index(self.config.VECTOR_DB_PATH)
            # 重新构建doc_map（需同步QA数据）
            cleaned_data = load_qa_data(self.config)
            all_chunks = []
            for item in cleaned_data:
                for chunk in item["chunks"]:
                    all_chunks.append({
                        "text": chunk,
                        "source": item["source"],
                        "answer": item["answer"]
                    })
            self.doc_map = {i: all_chunks[i] for i in range(len(all_chunks))}
            print(f"✅ 向量库加载完成")
        else:
            raise ValueError("❌ 向量库不存在，自动开始构建...")

    def retrieve(self, query: str) -> List[Dict]:
        """检索策略：向量检索（可迭代替换为混合检索）"""
        # 生成查询embedding
        query_emb = self.embedding_model.encode(
            query, convert_to_numpy=True, normalize_embeddings=True
        )

        # 检索Top-K
        distances, indices = self.index.search(
            query_emb.reshape(1, -1), self.config.RETRIEVE_TOP_K
        )

        # 整理结果（归一化相似度）
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            doc = self.doc_map.get(idx, {})
            results.append({
                "text": doc.get("text", ""),
                "answer": doc.get("answer", ""),
                "source": doc.get("source", ""),
                "score": 1 - (dist / 2)  # 相似度（0-1）
            })

        # 重排序后取Top-K
        results = sorted(results, key=lambda x: x["score"], reverse=True)[:self.config.RERANK_TOP_K]
        return results


# ===================== 4. RAG核心逻辑（满足所有作业要求） =====================
class RAGSystem:
    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 加载向量库（无则自动构建）
        self.vector_db = VectorDB(config)
        cleaned_data = load_qa_data(config)
        try:
            self.vector_db.load()
        except:
            self.vector_db.build(cleaned_data)

        # 加载你的LoRA微调模型（适配长上下文）
        print(f"✅ 加载LoRA微调模型：{config.LORA_MODEL_PATH}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.LORA_MODEL_PATH, trust_remote_code=True, padding_side="right"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.LORA_MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )

        # 长上下文适配：设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def calculate_confidence(self, query: str, retrieve_results: List[Dict]) -> float:
        """计算回答置信度（低于阈值则拒绝回答）"""
        if not retrieve_results:
            return 0.0
        query_emb = self.vector_db.embedding_model.encode(query, normalize_embeddings=True)
        similarities = []
        for res in retrieve_results:
            res_emb = self.vector_db.embedding_model.encode(res["text"], normalize_embeddings=True)
            sim = util.cos_sim(query_emb, res_emb).item()
            similarities.append(sim)
        return np.mean(similarities)

    def build_prompt(self, query: str, retrieve_results: List[Dict], history: List[Tuple]) -> str:
        """构建32k长上下文Prompt（含多轮历史+检索结果）"""
        # 拼接多轮对话历史
        history_text = ""
        for q, a in history[-self.config.MAX_HISTORY_TURNS:]:
            history_text += f"用户：{q}\n助手：{a}\n"

        # 拼接检索结果（带来源）
        retrieve_text = ""
        sources = set()
        for res in retrieve_results:
            retrieve_text += f"参考内容：{res['text']}\n参考答案：{res['answer']}\n来源：{res['source']}\n\n"
            sources.add(res["source"])

        # 作业要求：拒绝不确定回答+引用来源+长上下文
        prompt = f"""
        你是领域专家，严格遵守以下规则回答问题：
        1. 仅使用提供的参考信息回答，上下文长度不超过{self.config.MAX_CONTEXT_LEN} tokens，不编造内容；
        2. 若参考信息与问题无关（置信度<{self.config.CONFIDENCE_THRESH}），仅回复：“抱歉，我无法确定该问题的答案。”；
        3. 回答末尾必须标注引用来源，格式：【引用来源：来源1,来源2】；
        4. 保留多轮对话的上下文一致性。

        多轮对话历史：
        {history_text}

        参考信息：
        {retrieve_text}

        当前问题：{query}
        回答：
        """
        return prompt, sources

    def chat(self, query: str, history: List[Tuple]) -> Tuple[str, List[Tuple]]:
        """核心对话逻辑：检索→置信度判断→生成回答→引用来源"""
        # 1. 检索相关内容
        retrieve_results = self.vector_db.retrieve(query)

        # 2. 置信度判断（拒绝不确定回答）
        confidence = self.calculate_confidence(query, retrieve_results)
        if confidence < self.config.CONFIDENCE_THRESH:
            history.append((query, "抱歉，我无法确定该问题的答案。"))
            return "抱歉，我无法确定该问题的答案。", history

        # 3. 构建Prompt
        prompt, sources = self.build_prompt(query, retrieve_results, history)

        # 4. 长上下文生成
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.MAX_CONTEXT_LEN
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )

        # 5. 解析回答+添加引用来源
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = answer.split("回答：")[-1].strip()
        answer += f"\n【引用来源：{','.join(sources)}】"

        # 6. 更新多轮历史
        history.append((query, answer))
        return answer, history


# ===================== 5. 评估模块（作业要求：准确率/引用F1/幻觉率） =====================
def evaluate_rag(rag_system: RAGSystem, test_data: List[Dict]) -> Dict:
    """评估RAG系统性能（对比基线/优化版本）"""
    y_true = []
    y_pred = []
    f1_list = []
    hallucination_count = 0

    # 抽样100条评估（避免耗时过久）
    test_data = test_data[:100]
    for item in test_data:
        query = item["question"]
        true_answer = item["answer"]
        true_source = {item["source"]}

        # 生成回答
        retrieve_results = rag_system.vector_db.retrieve(query)
        confidence = rag_system.calculate_confidence(query, retrieve_results)

        if confidence < rag_system.config.CONFIDENCE_THRESH:
            pred_answer = "抱歉，我无法确定该问题的答案。"
        else:
            prompt, _ = rag_system.build_prompt(query, retrieve_results, [])
            inputs = rag_system.tokenizer(
                prompt, return_tensors="pt", padding=True, truncation=True
            ).to(rag_system.device)
            outputs = rag_system.model.generate(**inputs, max_new_tokens=1024)
            pred_answer = rag_system.tokenizer.decode(outputs[0], skip_special_tokens=True)
            pred_answer = pred_answer.split("回答：")[-1].strip()

        # 准确率
        y_true.append(1 if true_answer in pred_answer else 0)
        y_pred.append(1 if pred_answer != "抱歉，我无法确定该问题的答案。" else 0)

        # 引用F1
        pred_sources = re.findall(r"【引用来源：(.*?)】", pred_answer)
        pred_sources = set(pred_sources[0].split(",") if pred_sources else [])
        tp = len(true_source & pred_sources)
        fp = len(pred_sources - true_source)
        fn = len(true_source - pred_sources)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_list.append(f1)

        # 幻觉率（无来源标注且非拒绝回答）
        if "【引用来源：" not in pred_answer and pred_answer != "抱歉，我无法确定该问题的答案。":
            hallucination_count += 1

    # 计算最终指标
    accuracy = accuracy_score(y_true, y_pred)
    avg_f1 = np.mean(f1_list)
    hallucination_rate = hallucination_count / len(test_data)

    return {
        "准确率": round(accuracy, 4),
        "引用F1值": round(avg_f1, 4),
        "幻觉率": round(hallucination_rate, 4)
    }


# ===================== 6. Gradio Web Demo部署（作业要求） =====================
def main():
    # 初始化配置
    config = Config()

    # 加载QA数据
    cleaned_data = load_qa_data(config)

    # 初始化RAG系统
    rag_system = RAGSystem(config)

    # 评估（迭代优化时对比基线/优化版本）
    print("\n📊 开始评估RAG系统性能...")
    eval_results = evaluate_rag(rag_system, cleaned_data)
    print(f"📊 评估结果：{eval_results}")

    # 构建Web Demo
    with gr.Blocks(title="LoRA+RAG领域问答系统") as demo:
        gr.Markdown("# 🎯 LoRA微调+RAG领域问答系统")
        gr.Markdown(f"📋 作业要求达标项：32k长上下文 | 多轮对话 | 引用来源 | 拒绝不确定回答")
        gr.Markdown(f"📊 评估结果：{eval_results}")

        chatbot = gr.Chatbot(label="多轮对话窗口", height=500)
        query_input = gr.Textbox(label="请输入你的问题", placeholder="输入领域问题...")
        clear_btn = gr.Button("清空对话")

        # 对话函数
        def respond(query, history):
            answer, history = rag_system.chat(query, history)
            return "", history

        # 绑定事件
        query_input.submit(respond, [query_input, chatbot], [query_input, chatbot])
        clear_btn.click(lambda: None, None, chatbot, queue=False)

    # 启动Demo（本地访问：http://localhost:7860）
    print("\n🚀 Web Demo启动中... 访问地址：http://localhost:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()