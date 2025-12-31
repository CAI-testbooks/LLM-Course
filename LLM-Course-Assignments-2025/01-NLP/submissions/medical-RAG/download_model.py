# download_via_modelscope.py
from modelscope import snapshot_download

print("🚀 正在通过 ModelScope (魔搭社区) 下载 BAAI/bge-m3 ...")

# 1. 指定下载目录为当前目录下的 ./BAAI_bge-m3
# cache_dir 会自动处理文件结构
model_dir = snapshot_download(
    'BAAI/bge-m3', 
    cache_dir='./', 
    revision='master'
)

print(f"✅ 下载成功！模型已保存在: {model_dir}")
# ModelScope 下载后的路径通常是 ./BAAI/bge-m3，我们需要确认一下