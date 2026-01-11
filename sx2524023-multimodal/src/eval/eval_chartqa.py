from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
from tqdm import tqdm

from ..inference import load_qwen2vl, generate_answer


def normalize(text: str) -> str:
    text = text.strip().lower()
    # Remove commas in numbers: 1,000 -> 1000
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove trailing punctuation
    text = text.strip().strip(".。!！?？\n\t ")
    return text


def exact_match(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_jsonl", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--lora_path", type=str, default="")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--out_path", type=str, default="outputs/eval_results.json")
    args = parser.parse_args()

    lora_path = args.lora_path.strip() or None
    loaded = load_qwen2vl(model_id=args.model_id, lora_path=lora_path, load_in_4bit=False)

    records = []
    with open(args.eval_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if args.max_samples and args.max_samples > 0:
        records = records[: args.max_samples]

    correct = 0
    outputs = []
    for ex in tqdm(records, total=len(records)):
        img = Image.open(ex["image"]).convert("RGB")
        q = ex["question"]
        gold = ex["answer"]

        pred = generate_answer(loaded, [img], q, max_new_tokens=128, temperature=0.0, top_p=1.0)
        ok = exact_match(pred, gold)
        correct += int(ok)

        outputs.append(
            {
                "image": ex["image"],
                "question": q,
                "gold": gold,
                "pred": pred,
                "em": ok,
            }
        )

    acc = correct / max(1, len(records))
    out = {
        "model_id": args.model_id,
        "lora_path": args.lora_path,
        "num_samples": len(records),
        "exact_match": acc,
        "details": outputs,
    }

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ Exact Match: {acc:.4f} ({correct}/{len(records)})")
    print(f"✅ Saved to: {out_path}")


if __name__ == "__main__":
    main()
