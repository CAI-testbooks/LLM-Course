from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
)

from peft import LoraConfig, get_peft_model


SYSTEM_PROMPT = "You are a helpful multimodal assistant. Answer clearly and accurately."


class JsonlVQADataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.items: List[dict] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.items.append(json.loads(line))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        ex = self.items[idx]
        img = Image.open(ex["image"]).convert("RGB")
        return {
            "image": img,
            "question": str(ex.get("question", "")).strip(),
            "answer": str(ex.get("answer", "")).strip(),
        }


def _find_sublist(haystack: List[int], needle: List[int]) -> int:
    if not needle or not haystack:
        return -1
    n = len(needle)
    last = -1
    for i in range(len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            last = i
    return last


@dataclass
class DataCollatorQwen2VL:
    processor: any
    response_template: str = "<|im_start|>assistant\n"

    def __post_init__(self):
        self.tokenizer = self.processor.tokenizer
        self.resp_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)
        self.image_pad_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>") or -1

    def __call__(self, batch: Sequence[dict]) -> Dict[str, torch.Tensor]:
        images = [ex["image"] for ex in batch]
        texts = []

        for ex in batch:
            conv = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": ex["question"]}]},
                {"role": "assistant", "content": [{"type": "text", "text": ex["answer"]}]},
            ]
            texts.append(self.processor.apply_chat_template(conv, add_generation_prompt=False))

        model_inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        input_ids = model_inputs["input_ids"]
        labels = input_ids.clone()

        for i in range(labels.size(0)):
            ids = input_ids[i].tolist()
            start = _find_sublist(ids, self.resp_ids)
            if start != -1:
                labels[i, : start + len(self.resp_ids)] = -100

        if self.image_pad_id >= 0:
            labels[labels == self.image_pad_id] = -100

        pad_id = self.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        model_inputs["labels"] = labels
        return model_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--eval_jsonl", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="outputs/lora")
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_id,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        torch_dtype=dtype,
    )

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = JsonlVQADataset(args.train_jsonl)
    eval_ds = JsonlVQADataset(args.eval_jsonl) if args.eval_jsonl else None
    collator = DataCollatorQwen2VL(processor)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps" if eval_ds else "no",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    trainer.floating_point_ops = lambda *args, **kwargs: 0
    trainer.train()

    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"✅ LoRA saved to {args.output_dir}")


if __name__ == "__main__":
    main()

