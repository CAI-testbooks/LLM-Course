"""
微调系统配置文件
"""

import os
from pathlib import Path
import yaml
import json


class Config:
    """配置文件管理"""

    def __init__(self, config_path: str = None):
        self.base_dir = Path("/home/Liyang/agent")

        # 路径配置
        self.paths = {
            # 气象数据
            "weather_csv": self.base_dir / "weather_process" / "Bias_correction_ucl.csv",
            "weather_features": self.base_dir / "weather_process" / "weather_features.csv",
            "weather_ts_samples": self.base_dir / "weather_process" / "time_series_samples.pkl",

            # 卫星数据
            "eurosat_root": self.base_dir / "EuroSAT_MS",
            "eurosat_features": self.base_dir / "features" / "eurosat" / "features",
            "eurosat_model": self.base_dir / "features" / "eurosat" / "models" / "best_model.pth",

            # 文本知识
            "knowledge_json": self.base_dir / "knowledge_base" / "knowledge_base.json",
            "knowledge_embeddings": self.base_dir / "knowledge_base" / "embeddings.npy",

            # 微调输出
            "finetune_output": self.base_dir / "finetuned_models",
            "logs": self.base_dir / "logs",
            "checkpoints": self.base_dir / "checkpoints"
        }

        # 创建目录
        for path in self.paths.values():
            if isinstance(path, Path):
                path.parent.mkdir(parents=True, exist_ok=True)

        # 土壤类型配置
        self.soil_config = {
            "num_classes": 6,
            "soil_types": ["Clay", "Sandy", "Loamy", "Peaty", "Chalky", "Saline"],
            "class_weights": [1.0, 1.0, 1.0, 0.8, 0.8, 0.6],  # 样本不平衡权重

            # EuroSAT类别到土壤类型的映射（根据你的领域知识调整）
            "eurosat_to_soil": {
                0: 1,  # AnnualCrop -> Sandy
                1: 2,  # Forest -> Loamy
                2: 1,  # HerbaceousVegetation -> Sandy
                3: -1,  # Highway -> 忽略
                4: -1,  # Industrial -> 忽略
                5: 0,  # Pasture -> Clay
                6: 0,  # PermanentCrop -> Clay
                7: -1,  # Residential -> 忽略
                8: 5,  # River -> Saline
                9: 5  # SeaLake -> Saline
            }
        }

        # 气象预测配置
        self.weather_config = {
            "window_size": 7,
            "forecast_horizon": 1,
            "input_features": 64,  # 气象特征维度
            "output_features": 2,  # Next_Tmax, Next_Tmin
            "hidden_size": 128,
            "num_layers": 2
        }

        # 知识库配置
        self.knowledge_config = {
            "embedding_dim": 384,
            "top_k": 5,
            "similarity_threshold": 0.7,
            "max_context_length": 2048
        }

        # 训练配置
        self.training_config = {
            "batch_size": 32,
            "epochs": 100,
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "patience": 20,
            "gradient_clip": 1.0,
            "early_stopping": True,

            # 学习率调度
            "scheduler": {
                "name": "cosine",
                "warmup_epochs": 5,
                "min_lr": 1e-6
            },

            # 混合精度训练
            "mixed_precision": True,
            "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
        }

        # 智能体配置
        self.agent_config = {
            "perception": {
                "confidence_threshold": 0.7,
                "top_k_predictions": 3
            },
            "analysis": {
                "temperature_thresholds": {
                    "heatwave": 35,
                    "extreme_heat": 40,
                    "cold_wave": 0
                },
                "risk_levels": ["低", "中", "高", "极高"]
            },
            "knowledge": {
                "cache_size": 100,
                "retrieval_strategy": "hybrid"  # hybrid, semantic, keyword
            },
            "decision": {
                "fusion_method": "weighted",  # weighted, attention, ensemble
                "confidence_threshold": 0.6
            }
        }

        # 日志配置
        self.logging_config = {
            "log_level": "INFO",
            "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "log_file": self.paths["logs"] / "finetune.log"
        }

        # 加载外部配置文件（如果存在）
        if config_path and os.path.exists(config_path):
            self._load_external_config(config_path)

    def _load_external_config(self, config_path: str):
        """加载外部配置文件"""
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                external_config = yaml.safe_load(f)
            elif config_path.endswith('.json'):
                external_config = json.load(f)
            else:
                return

            # 更新配置
            for key, value in external_config.items():
                if hasattr(self, key):
                    if isinstance(getattr(self, key), dict) and isinstance(value, dict):
                        getattr(self, key).update(value)
                    else:
                        setattr(self, key, value)

    def save(self, path: str = None):
        """保存配置"""
        if path is None:
            path = self.paths["finetune_output"] / "config.yaml"

        config_dict = {
            "paths": {k: str(v) for k, v in self.paths.items()},
            "soil_config": self.soil_config,
            "weather_config": self.weather_config,
            "knowledge_config": self.knowledge_config,
            "training_config": self.training_config,
            "agent_config": self.agent_config
        }

        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        print(f"配置已保存到: {path}")

    def get_device(self):
        """获取设备"""
        import torch
        device = self.training_config["device"]
        if device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def __str__(self):
        """字符串表示"""
        info = "=" * 60 + "\n"
        info += "配置信息\n"
        info += "=" * 60 + "\n"

        info += f"设备: {self.training_config['device']}\n"
        info += f"土壤类型数: {self.soil_config['num_classes']}\n"
        info += f"气象窗口大小: {self.weather_config['window_size']}\n"
        info += f"训练轮数: {self.training_config['epochs']}\n"
        info += f"批大小: {self.training_config['batch_size']}\n"

        info += "\n路径:\n"
        for key, path in self.paths.items():
            info += f"  {key}: {path}\n"

        return info


# 全局配置实例

config = Config()
