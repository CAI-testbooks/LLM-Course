# Qwen2-VL Medical VQA System (医疗多模态诊断助手)

本项目基于 Qwen2-VL-2B 模型，使用 VQA-RAD 医疗数据集进行 LoRA 微调，实现了对医学影像（如 X 光、CT）的自动识别与诊断报告生成。

## 📂 目录结构
* `D:\Qwen2-VL-2B`: 基座模型权重
* `saves/Custom/lora/...`: 微调后的 Adapter 权重
* `run_cmd_demo.bat`: **一键启动脚本 (One-click Demo)**

## 🚀 如何运行 (How to Run)

本项目已封装 Windows 一键启动脚本，无需手动配置复杂环境。

1. **启动程序**：
   找到项目根目录下的 **`run_cmd_demo.bat`** 文件，双击运行。

2. **等待加载**：
   命令行窗口会自动激活环境并加载模型，请耐心等待直到出现 `User:` 提示符。

3. **进行诊断 (Usage)**：
   本系统支持 **图片拖拽** 输入。
   * **步骤 A**: 用鼠标将图片文件拖入黑色命令行窗口（路径会自动填入）。
   * **步骤 B**: 补全指令标签 `<image>...</image>` 并输入问题。

   **示例输入格式 (Example):**
   ```text
   <image>D:\images\lung_xray.png</image> 请生成这份影像的诊断报告。