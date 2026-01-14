# # src/models/model_manager.py
# import os
# import sys
# import yaml
# import shutil
# from typing import Dict, Optional, Any, List

# # 检查是否有足够的磁盘空间
# def check_disk_space(path: str, required_gb: float) -> bool:
#     """检查指定路径的磁盘空间"""
#     try:
#         total, used, free = shutil.disk_usage(path)
#         free_gb = free / (1024**3)  # 转换为GB
#         print(f"📊 {path} 可用空间: {free_gb:.1f}GB, 需要: {required_gb}GB")
#         return free_gb >= required_gb
#     except Exception as e:
#         print(f"⚠️ 无法检查磁盘空间: {e}")
#         return False  # 如果无法检查，假设空间不足

# def get_best_cache_dir() -> str:
#     """获取最佳缓存目录"""
#     # 尝试的缓存路径，按优先级排序
#     cache_options = [
#         ("D:/huggingface_cache", "D盘缓存"),
#         ("E:/huggingface_cache", "E盘缓存"),
#         ("C:/Users/Administrator/.cache/huggingface", "默认缓存"),
#         (os.path.expanduser("~/.cache/huggingface"), "用户缓存")
#     ]
    
#     for cache_path, description in cache_options:
#         try:
#             # 创建目录
#             os.makedirs(cache_path, exist_ok=True)
#             print(f"✅ 使用{description}: {cache_path}")
#             return cache_path
#         except Exception as e:
#             print(f"❌ {description}不可用: {e}")
    
#     # 使用临时目录作为最后的选择
#     temp_dir = os.path.join(os.getcwd(), "temp_cache")
#     os.makedirs(temp_dir, exist_ok=True)
#     print(f"⚠️ 使用临时缓存: {temp_dir}")
#     return temp_dir

# class ModelManager:
#     """管理多个代码LLM模型"""
    
#     def __init__(self, config_path: str = "config.yaml"):
#         self.config = self._load_config(config_path)
#         self.models: Dict[str, Dict[str, Any]] = {}
#         self.current_model = None
        
#         # 设置缓存路径到D盘
#         self._setup_cache()
        
#         # 根据磁盘空间智能选择模型
#         self._select_best_model()
    
#     def _setup_cache(self):
#         """设置缓存目录"""
#         # 获取最佳缓存路径
#         self.cache_dir = get_best_cache_dir()
        
#         # 设置环境变量
#         os.environ['HF_HOME'] = self.cache_dir
#         os.environ['TRANSFORMERS_CACHE'] = os.path.join(self.cache_dir, "models")
#         os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(self.cache_dir, "hub")
#         os.environ['HF_DATASETS_CACHE'] = os.path.join(self.cache_dir, "datasets")
        
#         # 设置国内镜像（加速下载）
#         os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
#         # 创建子目录
#         for subdir in ["models", "hub", "datasets"]:
#             path = os.path.join(self.cache_dir, subdir)
#             os.makedirs(path, exist_ok=True)
        
#         print(f"📁 模型缓存目录: {self.cache_dir}")
    
#     def _load_config(self, config_path: str) -> Dict[str, Any]:
#         """加载配置"""
#         try:
#             with open(config_path, 'r', encoding='utf-8') as f:
#                 return yaml.safe_load(f)
#         except FileNotFoundError:
#             print(f"⚠️ 配置文件 {config_path} 未找到，使用默认配置")
#             return self._get_default_config()
#         except Exception as e:
#             print(f"❌ 加载配置文件失败: {e}")
#             return self._get_default_config()
    
#     def _get_default_config(self) -> Dict[str, Any]:
#         """获取默认配置"""
#         return {
#             'models': {
#                 'default': 'tiny-starcoder',
#                 'options': {
#                     'tiny-starcoder': {
#                         'name': 'bigcode/tiny_starcoder_py',
#                         'type': 'huggingface',
#                         'max_tokens': 512,
#                         'size_gb': 0.2
#                     },
#                     'simulated-model': {
#                         'name': 'simulated',
#                         'type': 'simulated',
#                         'max_tokens': 1024,
#                         'size_gb': 0
#                     }
#                 }
#             }
#         }
    
#     def _select_best_model(self):
#         """根据磁盘空间智能选择最佳模型"""
#         if not self.config:
#             return
        
#         # 获取所有模型选项
#         models = self.config.get('models', {}).get('options', {})
#         if not models:
#             return
        
#         # 检查D盘空间
#         cache_drive = os.path.splitdrive(self.cache_dir)[0]
#         if not cache_drive:
#             cache_drive = "C:"  # 默认C盘
        
#         available_models = []
        
#         for model_name, model_config in models.items():
#             required_gb = model_config.get('size_gb', 999)
            
#             # 对于模拟模型，总是可用
#             if model_config.get('type') == 'simulated':
#                 available_models.append((model_name, model_config))
#                 continue
            
#             # 检查磁盘空间
#             if check_disk_space(cache_drive, required_gb * 1.5):  # 1.5倍安全系数
#                 available_models.append((model_name, model_config))
#                 print(f"✅ {model_name} 可用 ({required_gb}GB)")
#             else:
#                 print(f"❌ {model_name} 不可用 - 磁盘空间不足")
        
#         # 选择最小的可用模型
#         if available_models:
#             # 按大小排序
#             available_models.sort(key=lambda x: x[1].get('size_gb', 999))
#             best_model = available_models[0][0]
            
#             # 更新默认模型
#             if 'models' in self.config and 'default' in self.config['models']:
#                 old_default = self.config['models']['default']
#                 self.config['models']['default'] = best_model
#                 print(f"🔄 自动选择模型: {best_model} (原默认: {old_default})")
#         else:
#             print("⚠️ 没有可用模型，将使用模拟模式")
#             if 'models' in self.config:
#                 self.config['models']['default'] = 'simulated-model'
    
#     def get_model(self, model_name: Optional[str] = None) -> Dict[str, Any]:
#         """获取模型，带有自动降级功能"""
#         if not model_name:
#             model_name = self.config.get('models', {}).get('default', 'simulated-model')
        
#         # 如果已经加载过，直接返回
#         if model_name in self.models:
#             return self.models[model_name]
        
#         # 获取模型配置
#         model_config = self.config.get('models', {}).get('options', {}).get(model_name)
#         if not model_config:
#             print(f"❌ 模型 {model_name} 配置不存在")
#             return self._get_simulated_model()
        
#         print(f"🤖 正在加载模型: {model_name}")
#         print(f"   模型大小: {model_config.get('size_gb', '未知')}GB")
#         print(f"   缓存位置: {self.cache_dir}")
        
#         # 如果是模拟模型，直接返回
#         if model_config.get('type') == 'simulated':
#             return self._get_simulated_model()
        
#         # 尝试加载真实模型
#         try:
#             return self._load_real_model(model_name, model_config)
#         except Exception as e:
#             print(f"❌ 加载模型 {model_name} 失败: {e}")
            
#             # 尝试降级到更小的模型
#             return self._fallback_to_smaller_model(model_name)
    
#     def _load_real_model(self, model_name: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
#         """加载真实模型"""
#         # 延迟导入，避免在没有安装torch时出错
#         try:
#             import torch
#             from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
#         except ImportError as e:
#             print(f"❌ 缺少依赖: {e}")
#             raise
        
#         # 检查磁盘空间
#         required_gb = model_config.get('size_gb', 999)
#         cache_drive = os.path.splitdrive(self.cache_dir)[0] or "C:"
        
#         if not check_disk_space(cache_drive, required_gb * 1.2):  # 20%额外空间
#             raise OSError(f"磁盘空间不足，需要{required_gb}GB")
        
#         # 加载tokenizer
#         tokenizer = AutoTokenizer.from_pretrained(
#             model_config['name'],
#             trust_remote_code=True,
#             padding_side="left",
#             cache_dir=os.path.join(self.cache_dir, "models")
#         )
        
#         # 配置量化（节省内存）
#         quantization_config = None
#         if torch.cuda.is_available():
#             print("✅ 检测到GPU，使用4-bit量化")
#             quantization_config = BitsAndBytesConfig(
#                 load_in_4bit=True,
#                 bnb_4bit_compute_dtype=torch.float16,
#                 bnb_4bit_quant_type="nf4",
#                 bnb_4bit_use_double_quant=True,
#             )
        
#         # 加载模型
#         model = AutoModelForCausalLM.from_pretrained(
#             model_config['name'],
#             quantization_config=quantization_config,
#             torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#             device_map="auto" if torch.cuda.is_available() else None,
#             trust_remote_code=True,
#             use_safetensors=True,
#             cache_dir=os.path.join(self.cache_dir, "models")
#         )
        
#         # 确保pad_token设置
#         if tokenizer.pad_token is None:
#             tokenizer.pad_token = tokenizer.eos_token
        
#         model_info = {
#             'model': model,
#             'tokenizer': tokenizer,
#             'config': model_config,
#             'simulated': False
#         }
        
#         self.models[model_name] = model_info
#         self.current_model = model_name
        
#         print(f"✅ 模型 {model_name} 加载成功")
        
#         # 打印模型信息
#         total_params = sum(p.numel() for p in model.parameters())
#         print(f"📊 模型参数: {total_params:,}")
#         if torch.cuda.is_available():
#             print(f"🎮 使用设备: GPU ({torch.cuda.get_device_name(0)})")
#         else:
#             print("💻 使用设备: CPU")
        
#         return model_info
    
#     def _fallback_to_smaller_model(self, failed_model: str) -> Dict[str, Any]:
#         """降级到更小的模型"""
#         print("🔄 尝试降级到更小的模型...")
        
#         # 获取所有模型，按大小排序
#         all_models = list(self.config.get('models', {}).get('options', {}).items())
#         all_models.sort(key=lambda x: x[1].get('size_gb', 999))
        
#         # 找到失败的模型位置
#         failed_index = next((i for i, (name, _) in enumerate(all_models) 
#                            if name == failed_model), -1)
        
#         if failed_index == -1:
#             print("❌ 无法找到失败的模型配置")
#             return self._get_simulated_model()
        
#         # 尝试更小的模型
#         for i in range(failed_index + 1, len(all_models)):
#             model_name, model_config = all_models[i]
            
#             # 跳过模拟模型（最后的选择）
#             if model_config.get('type') == 'simulated':
#                 continue
            
#             print(f"🔄 尝试加载: {model_name}")
#             try:
#                 return self._load_real_model(model_name, model_config)
#             except Exception as e:
#                 print(f"❌ 加载 {model_name} 失败: {e}")
#                 continue
        
#         # 所有真实模型都失败，使用模拟模型
#         print("⚠️ 所有真实模型加载失败，使用模拟模型")
#         return self._get_simulated_model()
    
#     def _get_simulated_model(self) -> Dict[str, Any]:
#         """获取模拟模型"""
#         print("🎭 使用模拟模型（无需下载）")
        
#         model_info = {
#             'model': None,
#             'tokenizer': None,
#             'config': {
#                 'name': 'simulated-model',
#                 'type': 'simulated',
#                 'max_tokens': 1024,
#                 'size_gb': 0
#             },
#             'simulated': True
#         }
        
#         self.models['simulated-model'] = model_info
#         self.current_model = 'simulated-model'
        
#         return model_info
    
#     def list_available_models(self) -> List[str]:
#         """列出所有可用模型"""
#         models = self.config.get('models', {}).get('options', {})
#         return list(models.keys())
    
#     def get_model_info(self, model_name: str) -> Dict[str, Any]:
#         """获取模型信息"""
#         return self.config.get('models', {}).get('options', {}).get(model_name, {})

# src/models/model_manager.py
import os
import yaml
from typing import Dict, Optional, Any

class ModelManager:
    """模型管理器 - 专门使用小模型"""
    
    def __init__(self, config_path: str = "config.yaml"):
        # 直接设置缓存到D盘
        self._setup_disk_cache()
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 强制使用小模型，忽略配置文件
        self.config['models']['default'] = 'tiny-starcoder'
        
        self.models: Dict[str, Dict[str, Any]] = {}
        self.current_model = None
        
        print("🔧 已配置为使用小模型模式")
    
    def _setup_disk_cache(self):
        """设置缓存到D盘"""
        cache_dir = "D:/huggingface_cache_small"
        
        # 创建目录
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(f"{cache_dir}/models", exist_ok=True)
        os.makedirs(f"{cache_dir}/hub", exist_ok=True)
        
        # 设置环境变量
        os.environ['HF_HOME'] = cache_dir
        os.environ['TRANSFORMERS_CACHE'] = f"{cache_dir}/models"
        os.environ['HUGGINGFACE_HUB_CACHE'] = f"{cache_dir}/hub"
        
        # 设置国内镜像（加速下载）
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
        print(f"📁 缓存目录: {cache_dir}")
        print("🌐 使用国内镜像加速")
    
    def _load_config(self, config_path: str):
        """加载配置，但强制使用小模型"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except:
            # 如果配置文件不存在，使用默认配置
            config = self._get_default_config()
        
        # 确保配置中有tiny-starcoder
        if 'tiny-starcoder' not in config.get('models', {}).get('options', {}):
            config['models']['options']['tiny-starcoder'] = {
                'name': 'bigcode/tiny_starcoder_py',
                'type': 'huggingface',
                'max_tokens': 512,
                'size_gb': 0.2
            }
        
        return config
    
    def _get_default_config(self):
        """获取默认配置（小模型）"""
        return {
            'models': {
                'default': 'tiny-starcoder',
                'options': {
                    'tiny-starcoder': {
                        'name': 'bigcode/tiny_starcoder_py',
                        'type': 'huggingface',
                        'max_tokens': 512,
                        'size_gb': 0.2
                    },
                    'simulated-model': {
                        'name': 'simulated',
                        'type': 'simulated',
                        'max_tokens': 1024,
                        'size_gb': 0
                    }
                }
            }
        }
    
    def get_model(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """获取模型 - 强制使用小模型"""
        # 强制使用tiny-starcoder
        model_name = 'tiny-starcoder'
        
        if model_name in self.models:
            return self.models[model_name]
        
        model_config = self.config['models']['options'][model_name]
        
        print(f"🚀 正在加载小模型: {model_name}")
        print(f"   模型大小: {model_config['size_gb']}GB")
        print("   下载很快，请稍候...")
        
        try:
            # 尝试导入transformers
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError:
            print("❌ 未安装transformers，使用模拟模式")
            return self._get_simulated_model()
        
        try:
            # 加载小模型
            tokenizer = AutoTokenizer.from_pretrained(
                model_config['name'],
                trust_remote_code=True,
                padding_side="left"
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_config['name'],
                torch_dtype=torch.float32,  # 使用float32，更稳定
                device_map="cpu",  # 使用CPU，避免GPU内存问题
                trust_remote_code=True
            )
            
            # 确保pad_token设置
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model_info = {
                'model': model,
                'tokenizer': tokenizer,
                'config': model_config,
                'simulated': False
            }
            
            self.models[model_name] = model_info
            self.current_model = model_name
            
            print(f"✅ 小模型加载成功！")
            print(f"📊 模型参数: {sum(p.numel() for p in model.parameters()):,}")
            
            return model_info
            
        except Exception as e:
            print(f"❌ 小模型加载失败: {e}")
            print("🔄 切换到模拟模式...")
            return self._get_simulated_model()
    
    def _get_simulated_model(self):
        """获取模拟模型"""
        print("🎭 使用模拟模型（离线模式）")
        
        model_info = {
            'model': None,
            'tokenizer': None,
            'config': {
                'name': 'simulated-model',
                'type': 'simulated',
                'max_tokens': 1024,
                'size_gb': 0
            },
            'simulated': True
        }
        
        self.models['simulated-model'] = model_info
        self.current_model = 'simulated-model'
        
        return model_info