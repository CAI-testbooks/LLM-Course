"""
配置文件
"""
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    """配置类"""
    
    # LLM配置
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4-turbo-preview")
    llm_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    llm_api_base: Optional[str] = os.getenv("LLM_API_BASE")
    
    # DSPy配置
    max_tokens: int = 2000
    temperature: float = 0.1
    
    # 数据配置
    data_path: Optional[str] = None
    output_dir: str = "results"
    random_state: int = 42
    
    # 分析配置
    correlation_threshold: float = 0.7
    outlier_threshold: float = 1.5  # IQR乘数
    missing_value_threshold: float = 0.3  # 30%以上缺失考虑删除
    
    # 模型配置
    test_size: float = 0.2
    cv_folds: int = 5
    n_trials: int = 50  # Optuna试验次数
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化配置"""
        if config_path and os.path.exists(config_path):
            self._load_from_file(config_path)
        
        # 设置API密钥
        if self.llm_provider == "openai" and not self.llm_api_key:
            self.llm_api_key = os.getenv("OPENAI_API_KEY")
        elif self.llm_provider == "anthropic" and not self.anthropic_api_key:
            self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    
    def _load_from_file(self, config_path: str):
        """从文件加载配置"""
        # 这里可以添加从JSON/YAML文件加载配置的逻辑
        pass