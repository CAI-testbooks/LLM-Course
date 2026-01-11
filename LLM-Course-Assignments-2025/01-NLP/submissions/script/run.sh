#!/bin/bash
set -e  # 遇到错误立即退出

STREAMLIT_PORT=8501

SCRIPT_DIR=$(cd $(dirname $0) && pwd)
# 项目根目录（scripts/的上层目录）
PROJECT_ROOT=$(cd $SCRIPT_DIR/../ && pwd)
# chat_web.py的相对路径（项目根目录 -> src/chat_web.py）
CHAT_WEB_PATH="$PROJECT_ROOT/src/chat_web.py"

info "===== 环境检查 ====="
# 检查Python
if ! command -v python &> /dev/null; then
    error "未找到Python，请先安装Python 3.8+"
fi

# 检查Streamlit
if ! python -c "import streamlit" 2>/dev/null; then
    info "自动安装Streamlit..."
    pip install streamlit
fi

# 检查chat_web.py是否存在
if [ ! -f "$CHAT_WEB_PATH" ]; then
    error "未找到chat_web.py，路径：$CHAT_WEB_PATH"
fi
info "运行文件检查通过：$CHAT_WEB_PATH"

streamlit run "$CHAT_WEB_PATH" --server.port $STREAMLIT_PORT

info "服务已停止"