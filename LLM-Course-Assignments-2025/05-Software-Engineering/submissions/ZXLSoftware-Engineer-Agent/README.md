# 🤖 PUding Agent (AI Software Engineer)

一个交互式、可调用本地工具的 AI 编程助手。它支持通过 LLM（DeepSeek/OpenAI/Qwen/Gemini）进行需求理解、代码生成、测试执行、文件读写与编辑、命令运行等操作，并提供 CLI 和 Web 两种交互界面。
Github仓库地址:https://github.com/Epiphanyi/ZXLSoftware-Engineer-Agent

**核心目标**
- 在一个安全可控的工作目录内，借助 AI 自动化完成编码任务
- 通过“函数调用”桥接 AI 与本地文件系统/命令行
- 支持多种主流模型与 OpenAI 兼容接口

## ✨ 功能特性
- **多端支持**：提供命令行 (CLI) 和 Web 界面
- **模型接入**：DeepSeek/OpenAI/Qwen（通过兼容接口），Google Gemini
- **文件操作**：读取、创建、批量创建、编辑、目录列出
- **命令执行**：运行脚本、测试、构建等命令
- **流式输出**：实时展示 LLM 的响应与工具调用
- **上下文管理**：将文件/目录加入对话上下文以便更精准的代码分析
- **安全限制**：路径校验、最大文件大小限制、二进制文件自动跳过

## 🧱 项目结构
```text
ZXLSoftware-Engineer-Agent/
├── puding_agent/           # [核心代码包]
│   ├── agent.py            # AI 核心逻辑 (GeminiEngineer)
│   ├── cli.py              # CLI 界面逻辑
│   ├── tools.py            # 工具函数 (文件/命令操作)
│   ├── utils.py            # 辅助工具
│   └── config.py           # 配置与提示词
├── static/                 # Web 静态资源
├── templates/              # Web 模板
├── run_cli.py              # CLI 启动入口
├── web_ui.py               # Web 启动入口
├── requirements.txt        # 依赖清单
├── setup.py                # 安装配置
├── install.sh              # Linux/macOS 安装脚本
├── run.bat                 # Windows 启动脚本
└── .env                    # 环境变量配置
```

## 🚀 快速开始

### 前置要求
- Python 3.11+
- 有效的 API Key (Gemini, OpenAI, DeepSeek, 或 Qwen)

### 安装

**Windows**
```batch
# 1. 创建并激活虚拟环境
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. 安装依赖
pip install -r requirements.txt
pip install -e .

# 3. 配置 .env
cp env.example .env
# 编辑 .env 填入 API Key
```

**Linux/macOS**
```bash
./install.sh
```

### 运行

**方式 1：Web 界面 (推荐)**
```bash
# Windows 直接运行 run.bat
./run.bat

# 或者手动运行
python web_ui.py
```
访问 http://127.0.0.1:5000/

**方式 2：命令行 (CLI)**
```bash
python run_cli.py

# 或者安装后直接使用命令
puding-agent
```

## ⚙️ 配置说明 (.env)
以下为不同提供商的典型配置方式：

**DeepSeek（推荐）**
- `LLM_PROVIDER=openai`
- `OPENAI_BASE_URL=https://api.deepseek.com/v1`
- `OPENAI_MODEL=deepseek-coder`
- `OPENAI_API_KEY=你的_DeepSeek_API_Key`

**OpenAI**
- `LLM_PROVIDER=openai`
- `OPENAI_BASE_URL=https://api.openai.com/v1`
- `OPENAI_MODEL=gpt-4o`
- `OPENAI_API_KEY=你的_OpenAI_API_Key`

**Gemini**
- `LLM_PROVIDER=gemini`
- `GEMINI_API_KEY=你的_Gemini_API_Key`
- `GEMINI_MODEL=gemini-2.0-flash`

## 🖥️ 交互用法 (CLI)
- 启动后，命令行提示符为：`User >`
- 可用指令：
  - `/add <file_path>`：将指定文件加入上下文
  - `/help`：显示帮助说明
  - `/clear`：清空会话历史
  - `/exit` 或 `/quit`：退出应用

## 🔧 工具能力
- `run_command(command)`：运行命令
- `read_file(file_path)`：读取文件
- `create_file(file_path, content)`：创建/覆盖文件
- `edit_file(file_path, old_str, new_str)`：内容替换
- `list_directory(dir_path)`：列出目录

## 📜 许可证
MIT License
