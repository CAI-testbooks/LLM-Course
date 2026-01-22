# eval_base.py
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def generate_answer(model, tokenizer, instruction, input_text="", max_new_tokens=512):
    prompt = f"{instruction}\n{input_text}".strip() if input_text else instruction
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            top_k=3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    generated = outputs[0][inputs['input_ids'].shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True)
    return answer.strip()

def main():
    BASE_MODEL_PATH = "/root/autodl-tmp/qwen/Qwen2___5-7B-Instruct"
    TEST_DATA_PATH = "/root/autodl-tmp/Medical-RAG/dataset/alpaca_formatted_test_data.json"
    OUTPUT_DIR = "/root/autodl-tmp/Medical-RAG/eval_results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading test data...")
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        max_memory={0: "22GiB"},  # 👈 关键：防止 offload
        low_cpu_mem_usage=True
    )

    results = []
    for item in tqdm(test_data, desc="Evaluating Base Model"):
        instruction = item["instruction"]
        input_text = item.get("input", "").strip()
        reference = item["output"].strip()

        answer = generate_answer(model, tokenizer, instruction, input_text)
        results.append({
            "instruction": instruction,
            "input": input_text,
            "reference": reference,
            "answer": answer
        })

    output_path = os.path.join(OUTPUT_DIR, "base_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ Base model evaluation saved to: {output_path}")

if __name__ == "__main__":
    main()