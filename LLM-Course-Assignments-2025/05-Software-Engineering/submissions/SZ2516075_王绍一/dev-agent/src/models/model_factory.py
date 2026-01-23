# src/models/model_factory.py
import os
from typing import Dict, Any, Optional

class ModelFactory:
    """模型工厂，支持多种代码LLM"""
    
    def __init__(self, cache_dir: str = "D:/huggingface_cache"):
        self.cache_dir = cache_dir
        os.environ['HF_HOME'] = cache_dir
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
        # 支持的模型配置
        self.model_configs = {
            # 超小模型 - 200MB
            "tiny_starcoder": {
                "name": "bigcode/tiny_starcoder_py",
                "description": "专为Python的小模型，200MB",
                "params": "164M",
                "size_gb": 0.2
            },
            # DeepSeek系列
            "deepseek-coder-1.3b": {
                "name": "deepseek-ai/deepseek-coder-1.3b-instruct",
                "description": "DeepSeek 1.3B参数代码模型",
                "params": "1.3B",
                "size_gb": 2.7
            },
            "deepseek-coder-6.7b": {
                "name": "deepseek-ai/deepseek-coder-6.7b-instruct",
                "description": "DeepSeek 6.7B参数代码模型",
                "params": "6.7B",
                "size_gb": 14
            },
            # Qwen系列
            "qwen-coder-1.5b": {
                "name": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                "description": "Qwen 1.5B参数代码模型",
                "params": "1.5B",
                "size_gb": 3
            },
            # CodeLlama
            "codellama-7b": {
                "name": "codellama/CodeLlama-7b-Instruct-hf",
                "description": "CodeLlama 7B参数模型",
                "params": "7B",
                "size_gb": 14
            }
        }
    
    def create_model(self, model_id: str, use_quantization: bool = True) -> Dict[str, Any]:
        """创建指定模型"""
        if model_id not in self.model_configs:
            raise ValueError(f"未知模型: {model_id}")
        
        config = self.model_configs[model_id]
        print(f"🚀 创建模型: {model_id}")
        print(f"📊 参数: {config['params']}, 大小: {config['size_gb']}GB")
        
        # 检查磁盘空间
        if not self._check_disk_space(config['size_gb']):
            print(f"⚠️ 磁盘空间不足，建议使用更小模型")
            # 自动降级
            return self._auto_downgrade(config['size_gb'])
        
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            
            # 加载tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                config["name"],
                trust_remote_code=True,
                padding_side="left"
            )
            
            # 量化配置
            quantization_config = None
            if use_quantization and torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            
            # 加载模型
            model = AutoModelForCausalLM.from_pretrained(
                config["name"],
                quantization_config=quantization_config,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                use_safetensors=True
            )
            
            # 设置pad_token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            print(f"✅ 模型 {model_id} 创建成功")
            
            return {
                "model": model,
                "tokenizer": tokenizer,
                "config": config,
                "model_id": model_id
            }
            
        except Exception as e:
            print(f"❌ 模型创建失败: {e}")
            print("🔄 使用模拟模型...")
            return self._create_simulated_model(model_id)
    
    def _check_disk_space(self, required_gb: float) -> bool:
        """检查磁盘空间"""
        import shutil
        
        try:
            total, used, free = shutil.disk_usage(self.cache_dir[:2])
            free_gb = free / (1024**3)
            print(f"📊 可用空间: {free_gb:.1f}GB, 需要: {required_gb}GB")
            return free_gb >= required_gb * 1.5  # 1.5倍安全系数
        except:
            return True  # 如果无法检查，假设空间足够
    
    def _auto_downgrade(self, required_gb: float) -> Dict[str, Any]:
        """自动降级到合适的模型"""
        # 按大小排序
        sorted_models = sorted(
            self.model_configs.items(),
            key=lambda x: x[1]["size_gb"]
        )
        
        for model_id, config in sorted_models:
            if config["size_gb"] < required_gb:
                print(f"🔄 自动降级到: {model_id}")
                return self.create_model(model_id, use_quantization=True)
        
        # 如果所有模型都太大，使用模拟模型
        print("⚠️ 所有真实模型都太大，使用模拟模型")
        return self._create_simulated_model("simulated")
    
    def _create_simulated_model(self, model_id: str) -> Dict[str, Any]:
        """创建模拟模型"""
        print("🎭 创建模拟模型（离线模式）")
        
        return {
            "model": None,
            "tokenizer": None,
            "config": {
                "name": "simulated",
                "description": "模拟模型，无需下载",
                "params": "0",
                "size_gb": 0
            },
            "model_id": "simulated",
            "simulated": True
        }
    
    def list_available_models(self) -> Dict[str, Dict]:
        """列出所有可用模型信息"""
        return self.model_configs