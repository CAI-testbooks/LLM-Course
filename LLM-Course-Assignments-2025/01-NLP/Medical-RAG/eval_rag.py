import os
import json
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ===================== 配置区 (和你的RAG主代码完全一致，不要修改) =====================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
MODEL_NAME = "/root/autodl-tmp/qwen/Qwen2___5-7B-Instruct"
EMBEDDING_MODEL = "BAAI/bge-m3"
VECTOR_DB_PATH = "/root/autodl-tmp/Medical-RAG/chroma_db_medical"
TEST_DATA_PATH = "/root/autodl-tmp/Medical-RAG/dataset/alpaca_formatted_test_data.json"
OUTPUT_DIR = "/root/autodl-tmp/Medical-RAG/eval_results"
OUTPUT_FILE = "rag_results.json"  # RAG评估文件最终名称

# ===================== 导入RAG所需依赖 =====================
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ===================== 初始化完整的RAG系统 (复刻你的原版代码，无修改) =====================
def initialize_rag_chain():
    """初始化和主代码一模一样的RAG链，无UI相关代码"""
    # 1. 初始化嵌入模型
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # 2. 加载本地已构建好的向量库（必须提前运行主代码构建完成）
    vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 3. 加载分词器和大模型 (和主代码配置完全一致)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 4. 创建生成pipeline (和主代码配置完全一致)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.95,
        repetition_penalty=1.1,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        clean_up_tokenization_spaces=True,
        early_stopping=True
    )

    # 5. 封装LLM和Prompt模板 (和主代码完全一致)
    llm_pipeline = HuggingFacePipeline(pipeline=pipe)
    llm = ChatHuggingFace(llm=llm_pipeline, tokenizer=tokenizer, streaming=False)
    
    template = """
    你是一个专业的医学助手。
    回答要求：1. 条理清晰；2. 禁止重复表述；3. 生成答案时，不做冗余推理
    如果不知道，请直接说"根据现有医学资料，我无法提供确切答案，建议咨询专业医生"。

    医学知识：
    {context}

    用户问题：
    {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 6. 构建最终RAG链
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("✅ RAG系统初始化完成，向量库+大模型加载成功")
    return rag_chain

# ===================== 批量生成RAG答案 =====================
def generate_rag_answer(rag_chain, instruction, input_text=""):
    query = f"{instruction}\n{input_text}".strip() if input_text else instruction
    full_output = rag_chain.invoke(query)
    
    # 提取 assistant 的回答部分
    if "<|im_start|>assistant" in full_output:
        answer = full_output.split("<|im_start|>assistant")[-1].strip()
        # 去掉可能残留的 <|im_end|>
        if "<|im_end|>" in answer:
            answer = answer.split("<|im_end|>")[0].strip()
        return answer
    else:
        return full_output.strip()

# ===================== 主评估流程 =====================
def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    # 1. 加载测试集数据
    print(f"📄 加载测试集数据: {TEST_DATA_PATH}")
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"✅ 共加载 {len(test_data)} 条测试数据")

    # 2. 初始化RAG链
    rag_chain = initialize_rag_chain()

    # 3. 批量推理 + 结果保存
    results = [] 
    print("\n🚀 开始执行RAG批量评估（检索知识库+生成答案）...")
    for item in tqdm(test_data, desc="RAG评估进度", ncols=100):
        instruction = item["instruction"]
        input_text = item.get("input", "").strip()
        reference = item["output"].strip()

        answer = generate_rag_answer(rag_chain, instruction, input_text)
        results.append({
            "instruction": instruction,
            "input": input_text,
            "reference": reference,
            "answer": answer
        })

    # 4. 保存最终的RAG评估文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 评估完成！RAG评估文件已保存至: {output_path}")
    print(f"📊 文件包含 {len(results)} 条数据，字段与base评估完全一致！")

if __name__ == "__main__":
    main()