import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('Qwen/Qwen3-4B-Instruct', cache_dir='/root/autodl-tmp', revision='master')
