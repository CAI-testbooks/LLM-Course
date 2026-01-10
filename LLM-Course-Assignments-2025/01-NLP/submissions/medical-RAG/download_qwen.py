from modelscope import snapshot_download

print("🚀 正在通过 ModelScope 下载 Qwen/Qwen2.5-7B-Instruct ...")
print("⚠️ 注意：模型约 15GB，请确保磁盘空间充足。下载可能需要几分钟。")

model_dir = snapshot_download(
    'Qwen/Qwen2.5-7B-Instruct', 
    cache_dir='./', 
    revision='master'
)

print(f"✅ 下载成功！模型路径: {model_dir}")
# 通常路径为: ./Qwen/Qwen2.5-7B-Instruct