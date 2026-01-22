@echo off
title Qwen2-VL Medical VQA Demo (CLI Mode)
color 0A

echo ================================================================
echo       Qwen2-VL 医疗多模态诊断系统 (Medical VQA Assistant)
echo       [Local Run Mode]
echo ================================================================

:: ==============================================================================
::  ?? 运行前配置 (CONFIGURATION)
::  如果您在另一台电脑运行，请根据实际情况修改下面的路径
:: ==============================================================================

:: 1. 项目根目录 (你的 LLaMA-Factory 在哪？)
set PROJECT_ROOT=D:\damoxing\LLaMA-Factory

:: 2. 基座模型路径
set MODEL_PATH=D:\Qwen2-VL-2B

:: 3. LoRA 权重路径 (相对于项目根目录，或者绝对路径)
set ADAPTER_PATH=saves/Custom/lora/train_2026-01-09-09-49-47

:: 4. Conda 环境激活脚本
set CONDA_ACTIVATE=D:\anaconda3\Scripts\activate.bat
set ENV_NAME=qwen_env

:: ==============================================================================
::  ?? 启动流程 (STARTUP)
:: ==============================================================================

:: 1. 激活环境
echo [1/3] Activating environment...
call "%CONDA_ACTIVATE%" %ENV_NAME%

:: 2. 进入项目目录 (关键步骤！之前报错就是因为缺了这个)
echo [2/3] Entering project directory: %PROJECT_ROOT%
cd /d "%PROJECT_ROOT%"

:: 3. 启动模型
echo [3/3] Loading model...
echo.
echo [Tip] 等待出现 "User:" 提示符后，拖入图片即可。
echo.

llamafactory-cli chat ^
    --model_name_or_path "%MODEL_PATH%" ^
    --adapter_name_or_path "%ADAPTER_PATH%" ^
    --template qwen2_vl ^
    --trust_remote_code True

pause