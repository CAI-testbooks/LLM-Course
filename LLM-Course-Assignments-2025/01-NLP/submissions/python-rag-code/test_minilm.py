# 测试1：基本功能验证
from sentence_transformers import SentenceTransformer
import torch

print("=== 第1步：检查环境 ===")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print("\n=== 第2步：下载并加载模型 ===")
print("正在下载 all-MiniLM-L6-v2（约80MB）...")

# 首次运行会自动下载模型
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

print("✅ 模型下载完成！")
print(f"模型维度: {model.get_sentence_embedding_dimension()}")

print("\n=== 第3步：测试编码 ===")
# 测试句子
sentences = [
    "How to read a file in Python?",
    "Python中如何读取文件？",
    "Using open() function to read files",
    "文件读取操作示例"
]

print(f"编码 {len(sentences)} 个句子...")
embeddings = model.encode(sentences)

print("✅ 编码完成！")
print(f"嵌入向量形状: {embeddings.shape}")  # 应该是 (4, 384)
print(f"每个向量维度: {embeddings.shape[1]}")

print("\n=== 第4步：计算相似度 ===")
import numpy as np

# 计算第一个和第二个句子的相似度
similarity = np.dot(embeddings[0], embeddings[1]) / (
    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
)
print(f"中英文问题相似度: {similarity:.4f}")

# 保存模型到本地（避免重复下载）
print("\n=== 第5步：保存模型到本地 ===")
model.save("./models/all-MiniLM-L6-v2")
print("✅ 模型已保存到: ./models/all-MiniLM-L6-v2")

print("\n🎉 所有测试通过！all-MiniLM-L6-v2 运行成功。")