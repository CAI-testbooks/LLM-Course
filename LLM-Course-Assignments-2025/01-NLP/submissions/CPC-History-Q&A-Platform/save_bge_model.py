import os
from sentence_transformers import SentenceTransformer

# 1. 配置国内镜像（解决网络问题）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 2. 定义模型名称和本地保存路径
MODEL_NAME = 'BAAI/bge-large-zh-v1.5'
LOCAL_MODEL_PATH = './models/bge-large-zh-v1.5'  # 自定义本地路径，可修改

# 3. 确保保存目录存在
os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)

# 4. 下载模型并保存到本地
print(f"开始下载模型 {MODEL_NAME} 到 {LOCAL_MODEL_PATH}...")
model = SentenceTransformer(MODEL_NAME)
model.save_pretrained(LOCAL_MODEL_PATH)

print(f"模型下载完成！已保存到：{LOCAL_MODEL_PATH}")

# 5. 验证：加载本地模型（可选）
test_model = SentenceTransformer(LOCAL_MODEL_PATH)
test_embedding = test_model.encode("测试文本")
print(f"本地模型验证成功！生成的embedding维度：{test_embedding.shape}")