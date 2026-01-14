# src/config/config_manager.py
import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class RetrievalStrategy(Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    RERANK = "rerank"


@dataclass
class ModelConfig:
    """模型配置"""
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    quantization: str = "bf16"

    # 生成参数
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1

    # 微调参数
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    learning_rate: float = 2e-4


@dataclass
class RetrievalConfig:
    """检索配置"""
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    top_k: int = 5
    rerank_top_n: int = 3
    chunk_size: int = 512
    chunk_overlap: int = 50
    hybrid_weights: Dict[str, float] = field(
        default_factory=lambda: {"dense": 0.5, "sparse": 0.5})


@dataclass
class DataConfig:
    """数据配置"""
    raw_documents_path: str = "./data/raw"
    processed_chunks_path: str = "./data/processed/chunks.json"
    vector_db_path: str = "./data/vector_db"
    fine_tune_data_path: str = "./data/fine_tune"
    evaluation_data_path: str = "./data/evaluation"

    languages: List[str] = field(default_factory=lambda: ["zh", "en"])
    extract_tables: bool = True
    extract_images: bool = False


@dataclass
class APIConfig:
    """API配置"""
    web_host: str = "0.0.0.0"
    web_port: int = 7860
    web_debug: bool = True

    fastapi_host: str = "0.0.0.0"
    fastapi_port: int = 8000
    fastapi_workers: int = 4


@dataclass
class EvaluationConfig:
    """评估配置"""
    metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "rouge", "bleu", "hallucination_rate", "citation_f1"
    ])
    benchmarks: List[str] = field(default_factory=lambda: [
                                  "cmedqa", "legalbench"])


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    file: str = "./outputs/logs/app.log"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class RAGConfig:
    """RAG系统完整配置"""
    system_name: str = "rtca-rag-system"
    version: str = "1.0.0"
    environment: str = "development"

    model: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    data: DataConfig = field(default_factory=DataConfig)
    api: APIConfig = field(default_factory=APIConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, config_path: str) -> "RAGConfig":
        """从YAML文件加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        # 将嵌套字典转换为相应的配置类
        return cls(
            system_name=config_data.get("system", {}).get(
                "name", "rtca-rag-system"),
            version=config_data.get("system", {}).get("version", "1.0.0"),
            environment=config_data.get("system", {}).get(
                "environment", "development"),

            model=ModelConfig(**config_data.get("model", {})),
            retrieval=RetrievalConfig(**config_data.get("retrieval", {})),
            data=DataConfig(**config_data.get("data", {}).get("paths", {})),
            api=APIConfig(**config_data.get("api", {})),
            evaluation=EvaluationConfig(**config_data.get("evaluation", {})),
            logging=LoggingConfig(**config_data.get("logging", {}))
        )

    def to_yaml(self, config_path: str):
        """保存配置到YAML文件"""
        config_dict = {
            "system": {
                "name": self.system_name,
                "version": self.version,
                "environment": self.environment
            },
            "model": self.model.__dict__,
            "retrieval": self.retrieval.__dict__,
            "data": {
                "paths": self.data.__dict__,
                "processing": {
                    "languages": self.data.languages,
                    "extract_tables": self.data.extract_tables,
                    "extract_images": self.data.extract_images
                }
            },
            "api": self.api.__dict__,
            "evaluation": self.evaluation.__dict__,
            "logging": self.logging.__dict__
        }

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False,
                      allow_unicode=True)

    def update_from_env(self):
        """从环境变量更新配置"""
        import os

        # 更新设备配置
        if os.environ.get("USE_CPU"):
            self.model.device = "cpu"

        # 更新模型路径
        if os.environ.get("MODEL_PATH"):
            self.model.base_model = os.environ["MODEL_PATH"]

        # 更新API端口
        if os.environ.get("WEB_PORT"):
            self.api.web_port = int(os.environ["WEB_PORT"])

        if os.environ.get("API_PORT"):
            self.api.fastapi_port = int(os.environ["API_PORT"])
