#!/bin/bash
# setup.sh - CWRU数据分析代理系统安装脚本

echo "🚀 CWRU数据分析代理系统安装程序"
echo "===================================="

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python版本: $python_version"
if [[ "$python_version" < "3.11" ]]; then
    echo "❌ 需要Python 3.11或更高版本"
    exit 1
fi

# 创建虚拟环境
echo "创建虚拟环境..."
python3 -m venv venv

# 激活虚拟环境
if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "linux"* ]]; then
    source venv/bin/activate
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
fi

# 升级pip
echo "升级pip..."
pip install --upgrade pip

# 安装依赖
echo "安装依赖包..."
pip install -r requirements.txt

# 安装开发依赖（可选）
read -p "是否安装开发依赖？(y/n): " install_dev
if [[ "$install_dev" == "y" || "$install_dev" == "Y" ]]; then
    pip install pytest pytest-cov black flake8 mypy
fi

# 创建必要的目录
echo "创建项目目录..."
mkdir -p results/figures
mkdir -p results/models
mkdir -p logs
mkdir -p tests

# 设置环境变量
echo "设置环境变量..."
if [[ ! -f ".env" ]]; then
    cat > .env << EOF
# LLM配置
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
LLM_PROVIDER=openai
LLM_MODEL=gpt-4-turbo-preview

# 分析配置
RANDOM_STATE=42
TEST_SIZE=0.2
CV_FOLDS=5
EOF
    echo "✅ 已创建.env文件，请编辑并添加您的API密钥"
fi

# 创建示例配置文件
if [[ ! -f "config.yaml" ]]; then
    cat > config.yaml << EOF
# CWRU数据分析代理系统配置文件

llm:
  provider: "openai"
  model: "gpt-4-turbo-preview"
  temperature: 0.1
  max_tokens: 2000

data:
  path: "data_12k_10c.csv"
  output_dir: "results"
  random_state: 42

analysis:
  correlation_threshold: 0.7
  outlier_threshold: 1.5
  missing_value_threshold: 0.3

modeling:
  test_size: 0.2
  cv_folds: 5
  n_trials: 50

visualization:
  style: "seaborn-darkgrid"
  palette: "husl"
  figsize: [12, 8]
  dpi: 300
EOF
    echo "✅ 已创建config.yaml配置文件"
fi

echo ""
echo "🎉 安装完成！"
echo ""
echo "下一步："
echo "1. 编辑 .env 文件，添加您的API密钥"
echo "2. 激活虚拟环境:"
echo "   - Linux/Mac: source venv/bin/activate"
echo "   - Windows: venv\\Scripts\\activate"
echo "3. 运行分析: python run_analysis.py --data data_12k_10c.csv"
echo ""