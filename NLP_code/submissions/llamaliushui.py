import json
import torch
import os
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# ===================== 全局配置参数 =====================
# 输入测试集路径（所有模型共用）
INPUT_JSON_PATH = "/root/autodl-tmp/devdata1k.json"
# 生成配置（低temperature，可控随机性）
GENERATION_CONFIG = GenerationConfig(
    temperature=0.3,          # 低随机性（0.3-0.5为最优区间），既保证小幅差异又不偏离核心结果
    top_p=0.9,                # 核采样，保留90%概率的token（配合低temp增强稳定性）
    max_new_tokens=512,       # 最大生成token数
    do_sample=True,           # 开启采样（关键：引入可控随机性）
    eos_token_id=None,        # 自动识别结束符
    pad_token_id=None,        # 自动识别padding符
    repetition_penalty=1.05,  # 重复惩罚
)
# 使用GPU（如果可用）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========== 单次验证模式开关 ==========
RUN_ONLY_FIRST_EXPERIMENT = False  # 先验证1次，无误后改为False跑全量

# ===================== 待运行的模型列表 =====================
MODELS = [
    ("/root/autodl-tmp/llama-200", "llama-200"),
    ("/root/autodl-tmp/llama-400", "llama-400"),
    ("/root/autodl-tmp/llama-600", "llama-600"),
    ("/root/autodl-tmp/llama-800", "llama-800"),
]
# 实验次数
EXPERIMENT_TIMES = 3

# ===================== 加载模型和Tokenizer =====================
def load_model_and_tokenizer(model_path):
    """加载指定路径的模型和tokenizer"""
    print(f"\n=== 正在加载模型：{model_path} ===")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="right"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        ).eval()
        
        # 设置pad_token（Llama默认无pad_token）
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if model.config.pad_token_id is None:
            model.config.pad_token_id = model.config.eos_token_id
        
        print(f"模型 {model_path} 加载完成，使用设备：{DEVICE}")
        return tokenizer, model
    except Exception as e:
        print(f"加载模型 {model_path} 失败：{str(e)}")
        return None, None

# ===================== 生成函数（Llama格式） =====================
def generate_response(tokenizer, model, instruction, input_text):
    """适配Llama 3官方对话格式的生成函数"""
    prompt = f"""<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>

你需要严格按照指令要求回答问题。
<|eot_id|>
<|start_header_id|>user<|end_header_id|>

{instruction}

{input_text}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

"""
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    ).to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=GENERATION_CONFIG
        )
    
    generated_text = tokenizer.decode(
        outputs[0][len(inputs.input_ids[0]):],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    ).strip()
    
    return generated_text

# ===================== 单轮实验处理函数（新增随机种子） =====================
def run_single_experiment(model_path, model_name, exp_num):
    """运行单个模型的单次实验，为每次实验设置唯一随机种子"""
    # ========== 核心：为每次实验设置不同的随机种子 ==========
    # 种子值 = 模型名称哈希 + 实验序号（确保不同模型/实验的种子唯一）
    seed = hash(model_name) % 10000 + exp_num
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"本次实验随机种子：{seed}")
    
    # 生成输出文件路径
    output_file = f"/root/autodl-tmp/test_{model_name}_exp{exp_num}.json"
    
    # 防覆盖
    if os.path.exists(output_file):
        print(f"⚠️  输出文件 {output_file} 已存在，跳过本次实验")
        return
    
    # 加载模型
    tokenizer, model = load_model_and_tokenizer(model_path)
    if tokenizer is None or model is None:
        print(f"❌ 模型加载失败，跳过 {model_name} 第{exp_num}次实验")
        return
    
    # 读取数据集
    try:
        print(f"\n正在读取输入文件：{INPUT_JSON_PATH}")
        with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        if not isinstance(dataset, list):
            raise ValueError("输入JSON必须是列表格式")
    except Exception as e:
        print(f"❌ 读取输入文件失败：{str(e)}")
        return
    
    # 处理数据
    generated_dataset = []
    for idx, sample in enumerate(tqdm(dataset, desc=f"{model_name} 第{exp_num}次实验")):
        try:
            instruction = sample.get("instruction", "")
            input_text = sample.get("input", "")
            ground_truth = sample.get("output", "")
            
            # 生成回复（带随机性）
            generated_output = generate_response(tokenizer, model, instruction, input_text)
            
            new_sample = {
                "instruction": instruction,
                "input": input_text,
                "ground_truth": ground_truth,
                "generated_output": generated_output,
                "experiment_num": exp_num,
                "model_name": model_name,
                "seed": seed  # 记录种子，便于复现
            }
            generated_dataset.append(new_sample)
        except Exception as e:
            print(f"\n⚠️  处理第{idx}条数据时出错：{str(e)}")
            continue
    
    # 保存结果
    try:
        print(f"\n正在保存生成结果到：{output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(generated_dataset, f, ensure_ascii=False, indent=2)
        print(f"✅ {model_name} 第{exp_num}次实验完成，生成 {len(generated_dataset)} 条数据")
        print(f"📂 结果文件路径：{output_file}")
    except Exception as e:
        print(f"❌ 保存结果失败：{str(e)}")
    
    # 清理显存
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ===================== 批量运行主函数 =====================
def run_batch_experiments():
    print("=== 开始运行实验 ===")
    
    # 单次验证模式
    if RUN_ONLY_FIRST_EXPERIMENT:
        print(f"🔍 单次验证模式开启，仅运行第一个模型的第一次实验（1/15）")
        first_model_path, first_model_name = MODELS[0]
        first_exp_num = 1
        print(f"\n==================================================")
        print(f"开始运行：{first_model_name} - 第{first_exp_num}次实验")
        print(f"==================================================")
        run_single_experiment(first_model_path, first_model_name, first_exp_num)
        print("\n🎉 单次验证实验完成！")
        print(f"📌 结果文件：/root/autodl-tmp/test_{first_model_name}_exp{first_exp_num}.json")
        print(f"💡 验证后将 RUN_ONLY_FIRST_EXPERIMENT 改为 False 运行全量")
        return
    
    # 全量运行模式
    print(f"📦 全量运行模式开启")
    print(f"模型数量：{len(MODELS)} | 实验次数：{EXPERIMENT_TIMES} | 总计：{len(MODELS)*EXPERIMENT_TIMES}")
    
    for model_path, model_name in MODELS:
        for exp_num in range(1, EXPERIMENT_TIMES + 1):
            print(f"\n==================================================")
            print(f"开始运行：{model_name} - 第{exp_num}次实验")
            print(f"==================================================")
            run_single_experiment(model_path, model_name, exp_num)
    
    print("\n🎉 所有实验运行完成！")

if __name__ == "__main__":
    run_batch_experiments()