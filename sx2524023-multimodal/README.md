# Multimodal Vision-Language Assistant

课程作业：多模态视觉-语言助手（Qwen2-VL）

## Environment
- OS: Windows 11
- GPU: NVIDIA RTX 5070
- Python: Conda env: bo_env

## Base Evaluation (ChartQA)
- Model: Qwen/Qwen2-VL-2B-Instruct
- Split: val
- Samples: 200
- Metric: Exact Match (EM)
- Result: **0.23 (46/200)**

## Progress Log
- 2026-01-11: Base evaluation done (EM=0.23), demo is runnable.
- 2026-01-12: Attempted LoRA finetuning; training is slow and unstable under local network constraints.

## Reproduce

### 1) Prepare data (ChartQA)
```bash
python -m src.data.prepare --dataset chartqa --split val --max_samples 200
python -m src.data.prepare --dataset chartqa --split train --max_samples 5000


王睿@DESKTOP-487ANLA MINGW64 /d/multimodal (main)
$ cat > README.md << 'EOF'
# Multimodal Vision-Language Assistant

课程作业：多模态视觉-语言助手（Qwen2-VL）

## Environment
- OS: Windows 11
- GPU: NVIDIA RTX 5070
- Python: Conda env: bo_env

## Base Evaluation (ChartQA)
- Model: Qwen/Qwen2-VL-2B-Instruct
- Split: val
- Samples: 200
- Metric: Exact Match (EM)
- Result: **0.23 (46/200)**

## Progress Log
- 2026-01-11: Base evaluation done (EM=0.23), demo is runnable.
- 2026-01-12: Attempted LoRA finetuning; training is slow and unstable under local network constraints.

## Reproduce

### 1) Prepare data (ChartQA)
```bash
python -m src.data.prepare --dataset chartqa --split val --max_samples 200
python -m src.data.prepare --dataset chartqa --split train --max_samples 5000


