from modelscope import snapshot_download

print("🚀 正在通过 ModelScope (魔搭社区) 下载 BAAI/bge-m3 ...")

model_dir = snapshot_download(
    'BAAI/bge-m3', 
    cache_dir='./', 
    revision='master'
)

print(f"✅ 下载成功！模型已保存在: {model_dir}")