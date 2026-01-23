# RAG系统使用说明 - 阿里云通义千问版

## 项目概述

这是一个基于LangChain和阿里云通义千问构建的RAG（检索增强生成）系统，专门针对Mental Health数据集进行优化。
系统支持多层级文本切片、语义检索和灵活的相似度计算。
数据集来自huggingface的PrinceAyush/Mental_Health_conv
## 核心特性

### 🔍 智能检索
- **多层级切片**：对questionTitle、questionText、answerText分别进行语义切片
- **加权相似度计算**：支持灵活的权重配置
- **自适应权重接口**：预留扩展接口，支持动态权重调整

### 🛠️ 模块化设计
- **查询重写接口**：支持查询扩展和优化
- **可配置检索参数**：title_weight、question_weight、answer_weight
- **完整的统计信息**：提供系统运行状态

### 📊 相似度计算公式

默认使用加权平均：
```
最终相似度 = 0.5 * questionTitle相似度 + 0. 35* questionText相似度+ 0. 15* AnswerText相似度
```

扩展为：
```
最终相似度 = w1 * questionTitle相似度 + w2 * questionText相似度 + w3 * answerText相似度
```

## 安装和使用

### 1. 安装依赖

```bash
# 使用通义千问版本依赖
pip install -r requirements.qwen.txt

# 或者单独安装阿里云相关依赖
pip install dashscope langchain langchain-community faiss-cpu numpy
```

### 2. 获取阿里云API密钥

1. 访问 [DashScope控制台](https://dashscope.console.aliyun.com/)
2. 登录阿里云账号
3. 创建DashScope应用
4. 获取API Key

### 3. 设置API密钥

```bash
# 设置环境变量
export DASHSCOPE_API_KEY=your_dashscope_api_key_here
export ALIBABA_CLOUD_REGION=cn-beijing
#set DASHSCOPE_API_KEY=your_dashscope_api_key_here
#set ALIBABA_CLOUD_REGION=cn-beijing
```

或创建 `.env.qwen` 文件：
```
DASHSCOPE_API_KEY=your_dashscope_api_key_here
ALIBABA_CLOUD_REGION=cn-beijing
```

### 4. 运行演示

```bash
# 基础演示
python demo_qwen.py

# 查看支持的模型
python demo_qwen.py models
```

### 5. 基本使用示例

```python
from rag_system_qwen import QwenRAGSystem, RetrievalConfig

# 创建RAG系统
rag = QwenRAGSystem(
    embedding_model="text-embedding-v1",
    llm_model="qwen-turbo",
    chunk_size=500,
    chunk_overlap=100,
    retrieval_config=RetrievalConfig(
        title_weight=0.5,
        question_weight=0.35,
        answer_weight=0.15
    )
)

# 加载数据
rag.load_and_process_data("path/to/your/data.json")

# 构建向量数据库
rag.build_vectorstore()

# 搜索相关文档
results = rag.search("什么是恐慌发作？", top_k=5)

# 生成回答
answer = rag.generate_response("什么是恐慌发作？", top_k=5)
```


## 核心类说明

### QwenRAGSystem
主要的RAG系统类，包含以下方法：

- `load_and_process_data()`: 加载和处理JSON数据
- `build_vectorstore()`: 构建向量数据库
- `search()`: 搜索相关文档
- `generate_response()`: 生成回答
- `get_retrieval_stats()`: 获取统计信息

### RetrievalConfig
检索配置类，用于控制相似度计算的权重：

### 扩展接口

#### 查询重写
```python
def _query_rewrite(self, query: str) -> str:
    # 实现查询扩展、同义词替换等逻辑
    return query.strip()
```

#### 自适应权重
```python
def _adaptive_similarity_weights(self, query: str, record: DataRecord) -> Tuple[float, float, float]:
    # 根据查询特征动态调整权重
    return self.retrieval_config.title_weight, \
           self.retrieval_config.question_weight, \
           self.retrieval_config.answer_weight
```

## 性能优化建议

### 1. 文本切片参数调优
- `chunk_size`: 控制切片大小，较小值提高精确度，较大值提高召回率
- `chunk_overlap`: 控制重叠大小，避免切片边界信息丢失

### 2. 权重配置优化
- questionTitle权重应该较高，因为标题最能概括内容
- questionText权重适中，提供详细信息
- answerText权重根据具体应用场景调整

### 3. 检索参数调优
- `top_k`: 控制候选文档数量，平衡准确性和计算成本
- 相似度阈值：过滤低相似度结果

## 注意事项

### API使用
1. **费用控制**: 阿里云按token计费，建议设置费用限额
2. **并发限制**: 注意API的并发调用限制
3. **地域选择**: 建议选择就近的数据中心区域

### 性能考虑
1. **内存使用**: 大数据集可能需要较多内存，建议分批处理
2. **缓存策略**: 可以对embedding结果进行缓存
3. **异步处理**: 考虑使用异步API调用提高效率

### 数据格式
- 输入数据必须包含questionTitle、questionText、answerText字段
- 支持UTF-8编码的JSON格式

## 故障排除

### 常见问题

1. **API密钥错误**
   - 检查DASHSCOPE_API_KEY环境变量是否正确设置
   - 验证API密钥是否有效且有足够权限

2. **网络连接问题**
   - 检查网络连接
   - 确认防火墙设置允许访问阿里云服务

3. **内存不足**
   - 减少chunk_size参数
   - 考虑分批处理大数据集

4. **检索效果不佳**
   - 调整文本切片参数
   - 优化权重配置
   - 检查数据质量

### 错误代码处理

系统会捕获并处理以下常见错误：
- `InvalidAPIKeyError`: API密钥无效
- `RateLimitExceededError`: 超出速率限制
- `QuotaExceededError`: 超出配额限制
- `NetworkError`: 网络连接问题

### 日志和调试

系统提供详细的日志输出，包括：
- 数据加载进度
- 向量数据库构建状态
- 检索结果统计
- 错误信息和警告

## 扩展功能

系统预留了以下扩展接口：

1. **多模态支持**: 可扩展支持图像、音频等多媒体内容
2. **知识图谱集成**: 可结合知识图谱提升检索效果
3. **用户个性化**: 支持基于用户历史的个性化检索
4. **实时更新**: 支持动态添加新数据到知识库

## 成本优化建议

### 1. 模型选择策略
- 开发阶段使用`qwen-turbo`降低成本
- 生产环境根据需求选择合适的模型

### 2. 缓存策略
- 缓存query embedding结果
- 缓存LLM响应结果

### 3. 批量处理
- 批量处理embedding请求
- 合理设置batch_size

### 4. 监控和告警
- 监控API调用量
- 设置费用预警

## 许可证

本项目基于MIT许可证开源。

## 技术支持

- [DashScope官方文档](https://dashscope.console.aliyun.com/document)
- [LangChain文档](https://python.langchain.com/)
- [问题反馈](./issues)