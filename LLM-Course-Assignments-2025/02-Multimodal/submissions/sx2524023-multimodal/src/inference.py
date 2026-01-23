from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
)

from peft import PeftModel

from .utils import get_device


@dataclass
class LoadedModel:
    model_id: str
    model: Qwen2VLForConditionalGeneration
    processor: any
    device: str
    dtype: torch.dtype


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float16":
        return torch.float16
    return torch.float32


def load_qwen2vl(
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
    lora_path: Optional[str] = None,
    load_in_4bit: bool = True,
    trust_remote_code: bool = True,
) -> LoadedModel:
    """
    Load Qwen2-VL model + processor.

    - load_in_4bit is recommended for consumer GPUs.
    - lora_path: path to a PEFT LoRA adapter folder (output of our training script).
    """
    dev = get_device(prefer_cuda=True)
    dtype = _resolve_dtype(dev.dtype)

    quant_config = None
    if dev.is_cuda and load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto" if dev.is_cuda else None,
        torch_dtype=dtype if dev.is_cuda else torch.float32,
        quantization_config=quant_config,
        trust_remote_code=trust_remote_code,
    )

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)

    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
        # For inference it's nice to merge, but merging is optional and costs memory.
        # model = model.merge_and_unload()

    if not dev.is_cuda:
        model.to(dev.device)

    model.eval()
    return LoadedModel(model_id=model_id, model=model, processor=processor, device=dev.device, dtype=dtype)


def _build_conversation(prompt: str, n_images: int, system_prompt: Optional[str] = None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})

    content = []
    for _ in range(n_images):
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt})

    messages.append({"role": "user", "content": content})
    return messages


@torch.inference_mode()
def generate_answer(
    loaded: LoadedModel,
    images: List[Image.Image],
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.9,
    system_prompt: Optional[str] = "You are a helpful multimodal assistant. Answer clearly and accurately.",
) -> str:
    """Generate answer given 1+ images and a text prompt."""
    if not images:
        raise ValueError("images must be a non-empty list")

    processor = loaded.processor
    model = loaded.model

    conversation = _build_conversation(prompt=prompt, n_images=len(images), system_prompt=system_prompt)

    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(
        text=[text_prompt],
        images=images,
        padding=True,
        return_tensors="pt",
    )

    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    # Generate
    gen_kwargs = dict(
        max_new_tokens=int(max_new_tokens),
        do_sample=float(temperature) > 0,
        temperature=float(temperature),
        top_p=float(top_p),
    )

    output_ids = model.generate(**inputs, **gen_kwargs)

    # Slice out the prompt tokens
    input_len = inputs["input_ids"].shape[1]
    gen_ids = output_ids[:, input_len:]
    out = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    return out.strip()
