"""
Juris-RAG 配置文件
法律领域RAG问答系统配置
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ==================== 路径配置 ====================
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = str(BASE_DIR / "data" / "raw")
DB_PATH = str(BASE_DIR / "data" / "vector_db")
EVAL_DATA_PATH = str(BASE_DIR / "data" / "eval")
REPORTS_PATH = str(BASE_DIR / "reports")

# ==================== API配置 ====================
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"

# ==================== 模型配置 ====================
# Embedding模型 - 使用BGE-M3，支持中文法律领域
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIMENSION = 1024  # BGE-M3的向量维度

# LLM模型 - 使用Qwen2.5-7B-Instruct，支持128K长上下文
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# 备选模型
# LLM_MODEL = "Qwen/Qwen2.5-14B-Instruct"
# LLM_MODEL = "Qwen/Qwen3-8B"  # 注意：此模型可能不可用

# ==================== RAG参数配置 ====================
# 文本分块参数
CHUNK_SIZE = 800  # 每个文本块的大小（增大以保持法条完整）
CHUNK_OVERLAP = 150  # 文本块之间的重叠

# 检索参数
RETRIEVAL_TOP_K = 8  # 检索返回的文档数量
RETRIEVAL_SCORE_THRESHOLD = 0.3  # 相似度阈值（适度降低以检索到法条）

# 混合检索权重：优先检索法条
STATUTE_BOOST = 1.5  # 法条文档的权重提升

# 重排序参数（预留，尚未在引擎中启用）
ENABLE_RERANK = False
RERANK_TOP_K = 3  # 重排序后保留的文档数量

# ==================== 长上下文配置 ====================
# Qwen3-8B支持128k上下文
MAX_CONTEXT_LENGTH = 128000
MAX_INPUT_LENGTH = 32768  # 用户输入的最大长度
MAX_HISTORY_TURNS = 15  # 保留的历史对话轮数

# ==================== LLM生成参数 ====================
LLM_TEMPERATURE = 0.1  # 法律场景需要严谨，温度设低
LLM_MAX_TOKENS = 1024  # 最大生成长度（优化：从2048改为1024，采用更精炼的回答格式）
LLM_TOP_P = 0.9

# ==================== 置信度与拒答配置 ====================
# 当检索结果相似度低于此阈值时，拒绝回答
CONFIDENCE_THRESHOLD = 0.35
# 不确定回答的提示语
UNCERTAIN_RESPONSE = """抱歉，根据现有法律数据库，我无法准确回答此问题。

可能的原因：
1. 该问题涉及的法律领域不在我的知识范围内
2. 问题描述不够明确，请尝试提供更多细节
3. 这可能是一个需要专业律师判断的复杂法律问题

建议：请咨询专业律师获取准确的法律意见。"""

# ==================== Reranker/判别器配置 ====================
ENABLE_RERANKER = True  # 是否启用LLM重排序
RERANKER_MODEL = LLM_MODEL  # 使用同一模型进行重排序
RERANKER_TOP_K = 3  # 重排序后保留的文档数量（优化：从5改为3）
RERANKER_THRESHOLD = 0.4  # 相关性阈值，低于此值视为不相关
ENABLE_HALLUCINATION_CHECK = True  # 是否启用幻觉检测（优化：改为选择性启用）

# ==================== 混合幻觉检测策略（方案D）====================
# 根据问题类型采用不同的检测方法，平衡速度和准确性
SELECTIVE_HALLUCINATION_CHECK = True  # 启用选择性检测
# 高风险问题（特殊情况、超范围）：使用LLM深度检测
HALLUCINATION_CHECK_HIGH_RISK = True  # 对高风险问题启用LLM检测
HALLUCINATION_CHECK_HIGH_RISK_USE_LLM = True  # 高风险问题使用LLM检测（准确但慢）
# 正常问题（常规刑法）：跳过检测以加快响应
HALLUCINATION_CHECK_NORMAL = False  # 对正常问题禁用检测：以加快响应
HALLUCINATION_CHECK_TIMEOUT = 5  # 幻觉检测超时（秒）

# ==================== 检索缓存配置 ====================
ENABLE_CACHE = True  # 是否启用检索缓存
CACHE_MAX_SIZE = 200  # LRU缓存最大条目数（优化：从100改为200）
CACHE_TTL_SECONDS = 7200  # 缓存过期时间（秒）（优化：从3600改为7200，2小时）

# ==================== 并行处理配置 ====================
ENABLE_PARALLEL_RETRIEVAL = True  # 是否启用并行检索（短期优化）
MAX_PARALLEL_WORKERS = 4  # 最大并行工作线程数
PARALLEL_RETRIEVAL_TIMEOUT = 10  # 并行检索超时（秒）

# ==================== 流式生成配置 ====================
ENABLE_STREAM_GENERATION = True  # 是否启用流式生成
STREAM_CHUNK_SIZE = 50  # 流式输出的token块大小

# ==================== 数据处理配置 ====================
# CAIL数据集加载限制（可通过环境变量覆盖）
CAIL_CASE_LIMIT = int(os.getenv("CAIL_CASE_LIMIT", "100000"))

# 法条分割模式
STATUTE_SEPARATORS = ["\n第", "\n\n", "\n", "。", "；"]

# ==================== Web应用配置 ====================
APP_TITLE = "Juris-RAG 法律智能问答系统"
APP_DESCRIPTION = """
**Juris-RAG** 是一个基于检索增强生成（RAG）技术的中文法律问答系统。

**功能特点：**
- 支持刑法法条和案例检索
- 多轮对话，理解上下文
- 提供引用来源，可追溯
- 对不确定的问题会明确告知

**使用提示：**
- 输入您的法律问题，系统会检索相关法条和案例
- 系统会标注回答的来源依据
- 如需了解具体案例，可以描述案情
"""

# ==================== 评估配置 ====================
# 为避免 L0 (RPM=1000, TPM=50000) 触发限流，默认单条评估一批
EVAL_BATCH_SIZE = int(os.getenv("EVAL_BATCH_SIZE", "1"))
EVAL_METRICS = ["accuracy", "citation_f1", "hallucination_rate", "relevance"]

# ==================== 向量化节流配置 ====================
# 可通过环境变量覆盖，避免触发RPM/TPM限制
EMBED_RPM_LIMIT = int(os.getenv("EMBED_RPM_LIMIT", "1000"))
EMBED_TPM_LIMIT = int(os.getenv("EMBED_TPM_LIMIT", "50000"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "20"))
EMBED_SLEEP_SECONDS = float(os.getenv("EMBED_SLEEP_SECONDS", "0.1"))
EMBED_MAX_RETRIES = int(os.getenv("EMBED_MAX_RETRIES", "5"))
EMBED_BACKOFF_SECONDS = float(os.getenv("EMBED_BACKOFF_SECONDS", "10"))
EMBED_BACKOFF_MAX_SECONDS = float(os.getenv("EMBED_BACKOFF_MAX_SECONDS", "120"))

# ==================== LLM 调用节流配置 ====================
# 适配 SiliconFlow L0 默认限额：RPM=1000, TPM=50000
LLM_RPM_LIMIT = int(os.getenv("LLM_RPM_LIMIT", "1000"))
LLM_TPM_LIMIT = int(os.getenv("LLM_TPM_LIMIT", "50000"))
LLM_MIN_INTERVAL = 60.0 / LLM_RPM_LIMIT if LLM_RPM_LIMIT > 0 else 0.0
