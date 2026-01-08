"""
针对1650Ti（4GB显存）的优化配置
"""
from sentence_transformers import SentenceTransformer
import torch


class OptimizedMiniLM:
    def __init__(self):
        self.model = None
        self.device = None

    def setup_with_gpu_check(self):
        """智能设备选择"""
        # 检查GPU
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9


            if total_memory >= 3.5:  # 1650Ti有4GB，保留500MB给系统
                # 使用GPU，但采用内存优化策略
                self.device = 'cuda'

                # 清理GPU缓存
                torch.cuda.empty_cache()

                # 获取当前显存使用情况
                allocated = torch.cuda.memory_allocated(0) / 1e9
                cached = torch.cuda.memory_reserved(0) / 1e9
                print(f"当前显存使用: {allocated:.2f}GB / {cached:.2f}GB")

            else:
                self.device = 'cpu'
        else:
            self.device = 'cpu'

        # 加载模型（自动下载如果不存在）

        # 关键配置：避免一次性占用太多显存
        model_kwargs = {
            'device': self.device,
        }

        self.model = SentenceTransformer(
            'sentence-transformers/all-MiniLM-L6-v2',
            device=self.device
        )

        return self.model

    def encode_with_memory_control(self, texts, batch_size=16):
        """
        分批编码，避免显存溢出
        Args:
            texts: 文本列表
            batch_size: 批大小（1650Ti建议8-16）
        """
        if not self.model:
            self.setup_with_gpu_check()

        print(f"处理 {len(texts)} 个文本，批大小: {batch_size}")

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # 显示进度
            print(f"  批次 {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")

            # 编码当前批次
            batch_embeddings = self.model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True  # 归一化，提升相似度计算
            )

            all_embeddings.append(batch_embeddings)

            # 如果是GPU，清理缓存
            if self.device == 'cuda':
                torch.cuda.empty_cache()

        # 合并所有批次的嵌入
        import numpy as np
        embeddings = np.vstack(all_embeddings)

        print(f" 编码完成！生成 {embeddings.shape[0]} 个嵌入向量")
        return embeddings

    def test_performance(self):
        """性能测试"""

        # 创建测试数据
        test_texts = [f"Python code example {i}: how to use functions and classes"
                      for i in range(50)]

        # 测试不同批大小
        batch_sizes = [4, 8, 16, 32]

        for batch_size in batch_sizes:
            print(f"\n测试批大小: {batch_size}")

            # 清理缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            import time
            start_time = time.time()

            embeddings = self.encode_with_memory_control(test_texts[:10], batch_size)

            elapsed = time.time() - start_time

            if self.device == 'cuda':
                peak_memory = torch.cuda.max_memory_allocated() / 1e9
                print(f"  峰值显存: {peak_memory:.3f}GB")

            print(f"  处理时间: {elapsed:.2f}秒")
            print(f"  平均每句: {elapsed / 10:.3f}秒")


# 运行测试
if __name__ == "__main__":

    # 创建优化实例
    optimizer = OptimizedMiniLM()

    # 设置模型
    model = optimizer.setup_with_gpu_check()

    # 简单测试
    test_sentences = [
        "How to install Python packages?",
        "What is a virtual environment?",
        "How to create a class in Python?",
        "Python装饰器的作用是什么？"
    ]

    print("\n=== 简单编码测试 ===")
    embeddings = optimizer.encode_with_memory_control(test_sentences, batch_size=8)

    print(f"\n嵌入向量示例（第一句的前10维）:")
    print(embeddings[0][:10])

    # 性能测试（可选）
    # optimizer.test_performance()
