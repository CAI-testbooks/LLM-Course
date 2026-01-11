from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import gradio as gr
from PIL import Image

from .inference import load_qwen2vl, generate_answer
from .video_utils import sample_video_frames


DEFAULT_MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2-VL-2B-Instruct")
DEFAULT_LORA_PATH = os.getenv("LORA_PATH", "")  # optional


def _load_images_from_files(files) -> List[Image.Image]:
    imgs: List[Image.Image] = []
    if not files:
        return imgs
    for f in files:
        path = None
        # Gradio can return FileData objects / dicts / strings depending on version
        if isinstance(f, str):
            path = f
        elif isinstance(f, dict):
            path = f.get("path") or f.get("name")
        else:
            path = getattr(f, "path", None) or getattr(f, "name", None)

        if not path:
            continue
        img = Image.open(path).convert("RGB")
        imgs.append(img)
    return imgs


def _video_to_frames(video, fps: float, max_frames: int):
    if not video:
        return []
    path = None
    if isinstance(video, str):
        path = video
    elif isinstance(video, dict):
        path = video.get("path") or video.get("name")
    else:
        path = getattr(video, "path", None) or getattr(video, "name", None)

    if not path:
        return []

    result = sample_video_frames(path, fps=fps, max_frames=max_frames)
    return result.frames


def build_demo():
    with gr.Blocks(title="Multimodal Vision-Language Assistant (Qwen2-VL)") as demo:
        gr.Markdown(
            """

# 多模态视觉-语言助手（示例：Qwen2-VL）

- 支持：**单图/多图** + 文本提问；或上传**视频**（自动抽帧）+ 文本提问  
- 说明：此 Demo 只是作业的 **前端交互** 部分；可以在代码仓库里继续做 LoRA 微调、评测与迭代。  
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                model_id = gr.Textbox(label="Model ID", value=DEFAULT_MODEL_ID)
                lora_path = gr.Textbox(label="LoRA Adapter Path (optional)", value=DEFAULT_LORA_PATH, placeholder="例如: outputs/lora-qwen2vl-chartqa")
                load_btn = gr.Button("加载/重载模型")

                imgs = gr.Files(label="上传图片（可多张）", file_types=["image"], file_count="multiple")
                vid = gr.Video(label="或上传视频（可选）")

                fps = gr.Slider(0.5, 5.0, value=1.0, step=0.5, label="视频抽帧 FPS（仅视频有效）")
                max_frames = gr.Slider(4, 32, value=16, step=1, label="最多抽取帧数（仅视频有效）")

                prompt = gr.Textbox(label="问题 / 指令", lines=3, placeholder="例如：这张图里有几条折线？最高值是多少？")
                max_new_tokens = gr.Slider(32, 512, value=256, step=8, label="max_new_tokens")
                temperature = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="top_p")

                run_btn = gr.Button("开始回答")

            with gr.Column(scale=1):
                status = gr.Markdown("模型未加载")
                output = gr.Textbox(label="模型回答", lines=18)

        state = gr.State(value=None)

        def do_load(mid: str, lp: str):
            lp = lp.strip()
            if lp == "":
                lp = None
            loaded = load_qwen2vl(model_id=mid.strip(), lora_path=lp, load_in_4bit=False)
            return loaded, f"✅ 已加载：{loaded.model_id}\n\n设备：{loaded.device} dtype：{loaded.dtype}"

        def do_run(loaded, files, video, q, fps, max_frames, max_new_tokens, temperature, top_p):
            if loaded is None:
                return "❌ 你还没加载模型，先点『加载/重载模型』", ""
            q = (q or "").strip()
            if not q:
                return "❌ 请输入问题/指令", ""

            images = _load_images_from_files(files)
            if not images:
                images = _video_to_frames(video, fps=float(fps), max_frames=int(max_frames))

            if not images:
                return "❌ 需要上传至少一张图片或一个视频", ""

            ans = generate_answer(
                loaded=loaded,
                images=images,
                prompt=q,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
            )
            return f"✅ 完成（输入图像数量：{len(images)}）", ans

        load_btn.click(do_load, inputs=[model_id, lora_path], outputs=[state, status])
        run_btn.click(do_run, inputs=[state, imgs, vid, prompt, fps, max_frames, max_new_tokens, temperature, top_p], outputs=[status, output])

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="127.0.0.1", server_port=int(os.getenv("PORT", "7860")))
