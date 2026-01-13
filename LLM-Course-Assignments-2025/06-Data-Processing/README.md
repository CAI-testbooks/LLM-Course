# Kaggle乳腺癌数据分析代理系统

## 项目概述

本项目是一个AI驱动的乳腺癌数据分析代理，专门用于分析Kaggle乳腺癌数据集。系统能够自动化完成从数据加载、清洗、探索性分析、特征工程、建模评估到报告生成的全流程。

## 核心功能

1. **全自动化分析流程**：从原始数据到洞察报告的一站式解决方案
2. **智能数据预处理**：自动检测和处理缺失值、异常值
3. **高级特征工程**：特征缩放、选择、重要性分析
4. **多模型对比**：集成多种机器学习算法，自动选择最佳模型
5. **可视化报告**：自动生成包含图表和分析的专业报告
6. **Web界面支持**：提供用户友好的Web上传和分析界面

## 技术栈

- **后端框架**：Flask
- **数据分析**：Pandas, NumPy
- **机器学习**：Scikit-learn
- **数据可视化**：Matplotlib, Seaborn
- **报告生成**：Jinja2, Markdown
- **环境管理**：Python 3.8+

## 安装步骤

### 1. 克隆项目
```bash
git clone <项目地址>
cd breast_cancer_kaggle_analysis
```

### 2. 创建虚拟环境（推荐）
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 准备数据
将您的Kaggle乳腺癌数据集CSV文件重命名为`data.csv`，并放置在`data/`目录下。

如果您的数据文件名不同，可以在运行程序时指定路径。

## 使用方法

### 方式一：命令行运行
```bash
# 使用默认数据文件（data/breast_cancer_kaggle.csv）
python main.py

# 指定自定义数据文件
python main.py /path/to/your/data.csv
```

### 方式二：Web应用运行
```bash
python web_app.py
```
然后访问 http://localhost:5000 使用Web界面。

## 项目结构

```
breast_cancer_kaggle_analysis/
├── README.md                    # 项目说明文档
├── requirements.txt             # Python依赖包列表
├── main.py                      # 命令行主程序入口
├── web_app.py                   # Web应用入口
├── config.py                    # 配置文件
├── setup_environment.py         # 环境配置脚本
├── agents/                      # 核心功能模块
│   ├── __init__.py
│   ├── data_loader_kaggle.py   # Kaggle数据加载器
│   ├── data_cleaner.py         # 数据清洗模块
│   ├── eda_analyzer.py         # EDA分析模块
│   ├── feature_engineer.py     # 特征工程模块
│   ├── model_builder.py        # 建模评估模块
│   ├── visualizer.py           # 可视化模块
│   └── report_generator.py     # 报告生成模块
├── data/                         # 数据目录
│   └── breast_cancer_kaggle.csv  # Kaggle乳腺癌数据集
├── templates/                   # HTML模板
│   └── index.html
├── uploads/                    # 上传文件临时存储
└── reports/                    # 生成的报告存储
```

## 配置文件说明

`config.py` 包含以下配置项：

```python
# 数据集配置
DATA_CONFIG = {
    'default_path': 'data/data.csv',  # 默认数据路径
    'id_column': 'id',  # ID列名
    'diagnosis_column': 'diagnosis',  # 诊断结果列名
    'diagnosis_mapping': {'B': 0, 'M': 1},  # 诊断结果映射
    'drop_columns': ['id', 'Unnamed: 32'],  # 需要删除的列
}
```

如果您的数据集结构不同，请相应修改这些配置。

## 分析流程

系统执行以下完整的分析流程：

1. **数据加载与预处理**
   - 自动识别诊断结果列
   - 转换分类变量为数值型
   - 删除无关列

2. **数据清洗**
   - 缺失值检测与处理
   - 异常值检测

3. **探索性数据分析**
   - 基本统计描述
   - 相关性分析
   - 分布分析

4. **特征工程**
   - 特征标准化
   - 特征选择（ANOVA，随机森林重要性）
   - PCA降维（可选）

5. **建模与评估**
   - 训练多个机器学习模型
   - 5折交叉验证
   - 模型性能对比

6. **报告生成**
   - 自动生成Markdown和HTML格式报告
   - 生成可视化图表

## 支持的机器学习模型

- Logistic Regression（逻辑回归）
- Decision Tree（决策树）
- Random Forest（随机森林）
- Gradient Boosting（梯度提升）
- Support Vector Machine（支持向量机）
- K-Nearest Neighbors（K近邻）

## 输出结果

分析完成后，系统在`reports/`目录下生成以下文件：

1. **报告文件**
   - `breast_cancer_kaggle_report_YYYYMMDD_HHMMSS.md` - Markdown格式报告
   - `breast_cancer_kaggle_report_YYYYMMDD_HHMMSS.html` - HTML格式报告（可在浏览器中打开）

2. **可视化图表**
   - `correlation_heatmap_YYYYMMDD_HHMMSS.png` - 特征相关性热力图
   - `distribution_特征名_YYYYMMDD_HHMMSS.png` - 重要特征分布图
   - `model_comparison_YYYYMMDD_HHMMSS.png` - 模型性能对比图

3. **数据文件**
   - `analysis_results_YYYYMMDD_HHMMSS.json` - 原始分析结果数据

## Web界面使用说明

1. 启动Web服务器后，访问 http://localhost:5000
2. 点击"选择CSV文件"上传您的乳腺癌数据集
3. 或点击"使用默认数据集"（如果已放置data/breast_cancer_kaggle.csv文件）
4. 等待分析完成（通常需要30-60秒）
5. 下载生成的报告文件

## 常见问题解决

### 1. 端口被占用
```bash
# 修改web_app.py中的端口号
app.run(debug=False, host='0.0.0.0', port=5001)
```

### 2. 内存不足
- 减少数据集大小
- 关闭其他应用程序
- 增加系统虚拟内存

### 3. 依赖安装失败
```bash
# 使用国内镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 4. Tkinter相关错误（已在代码中修复）
项目已配置使用matplotlib的Agg后端，避免在Web环境中出现Tkinter线程问题。

## 自定义和扩展

### 添加新的机器学习模型
在`agents/model_builder.py`的`train_models`方法中添加新的模型配置。

### 修改可视化样式
在`agents/visualizer.py`中调整matplotlib的样式设置。

### 扩展报告内容
在`agents/report_generator.py`中修改报告模板。

## 数据要求

系统期望的CSV数据格式：
- 包含一个诊断结果列（默认列名为'diagnosis'，取值为'M'和'B'）
- 包含多个数值型特征列
- 可以有ID列，系统会自动删除
- 建议使用Kaggle的威斯康星乳腺癌诊断数据集

## 性能优化

- 对于大型数据集，建议：
  1. 增加系统内存
  2. 减少特征数量
  3. 使用更简单的模型
  4. 分批处理数据


**免责声明**：本项目仅供学习和研究使用，生成的分析报告仅供参考，不能替代专业医疗诊断。实际医疗决策应结合临床医生专业判断和更多检查结果。