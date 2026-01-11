"""
知识库适应微调
微调检索模型以适应特定气候领域
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
import logging
from datetime import datetime

warnings.filterwarnings('ignore')

from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.datasets import SentencesDataset

from config import config


class KnowledgeDataset(Dataset):
    """知识库检索数据集"""

    def __init__(self, queries: List[str], documents: List[str],
                 labels: List[List[int]], query_embeddings: np.ndarray = None,
                 doc_embeddings: np.ndarray = None):
        """
        初始化数据集

        Args:
            queries: 查询文本列表
            documents: 文档文本列表
            labels: 每个查询的相关文档索引列表
            query_embeddings: 预计算的查询嵌入
            doc_embeddings: 预计算的文档嵌入
        """
        self.queries = queries
        self.documents = documents
        self.labels = labels
        self.query_embeddings = query_embeddings
        self.doc_embeddings = doc_embeddings

        # 验证数据
        assert len(queries) == len(labels), "查询和标签数量不一致"

        print(f"知识库数据集: {len(queries)} 查询, {len(documents)} 文档")
        print(f"平均相关文档数: {np.mean([len(l) for l in labels]):.2f}")

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        relevant_docs = self.labels[idx]

        # 随机选择一个相关文档
        if relevant_docs:
            pos_idx = np.random.choice(relevant_docs)
            pos_doc = self.documents[pos_idx]
        else:
            # 如果没有相关文档，随机选择一个
            pos_idx = np.random.randint(0, len(self.documents))
            pos_doc = self.documents[pos_idx]

        # 随机选择一个不相关文档
        neg_candidates = [i for i in range(len(self.documents))
                          if i not in relevant_docs]
        neg_idx = np.random.choice(neg_candidates) if neg_candidates else np.random.randint(0, len(self.documents))
        neg_doc = self.documents[neg_idx]

        return {
            'query': query,
            'positive': pos_doc,
            'negative': neg_doc,
            'query_idx': idx,
            'pos_idx': pos_idx,
            'neg_idx': neg_idx
        }

    def create_triplets(self) -> List[Tuple[str, str, str]]:
        """创建三元组数据"""
        triplets = []

        for idx in range(len(self.queries)):
            query = self.queries[idx]
            relevant_docs = self.labels[idx]

            if not relevant_docs:
                continue

            # 对每个相关文档，创建一个负样本
            for pos_idx in relevant_docs:
                pos_doc = self.documents[pos_idx]

                # 寻找负样本
                neg_candidates = [i for i in range(len(self.documents))
                                  if i not in relevant_docs]
                if neg_candidates:
                    neg_idx = np.random.choice(neg_candidates)
                    neg_doc = self.documents[neg_idx]

                    triplets.append((query, pos_doc, neg_doc))

        return triplets

    def create_triplet_examples(self) -> List[InputExample]:
        """创建三元组InputExample列表，用于sentence-transformers训练"""
        examples = []
        triplets = self.create_triplets()

        for query, pos_doc, neg_doc in triplets:
            # 创建InputExample，texts包含三个元素：[anchor, positive, negative]
            examples.append(InputExample(texts=[query, pos_doc, neg_doc]))

        return examples


class KnowledgeAdaptor:
    """知识库适应器"""

    def __init__(self):
        self.config = config
        self.device = self.config.get_device()

        # 模型
        self.model = None
        self.train_loss = None
        self.evaluator = None

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_ndcg': [],
            'val_map': [],
            'val_recall': []
        }

        # 输出目录
        self.output_dir = Path(self.config.paths["finetune_output"]) / "knowledge_adaptor"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 日志
        self._setup_logging()

        print(self.config)

    def _setup_logging(self):
        """设置日志"""
        import logging

        log_file = self.output_dir / "training.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger(__name__)

    def load_knowledge_base(self) -> Dict:
        """加载知识库"""
        print("加载知识库...")

        knowledge_path = Path(self.config.paths["knowledge_json"])

        if not knowledge_path.exists():
            raise FileNotFoundError(f"知识库文件不存在: {knowledge_path}")

        with open(knowledge_path, 'r', encoding='utf-8') as f:
            knowledge_base = json.load(f)

        # 加载嵌入向量
        embeddings_path = Path(self.config.paths["knowledge_embeddings"])
        if embeddings_path.exists():
            embeddings = np.load(embeddings_path)
        else:
            embeddings = None

        print(f"知识库条目: {len(knowledge_base.get('items', []))}")

        return {
            'knowledge_base': knowledge_base,
            'embeddings': embeddings
        }

    def create_synthetic_queries(self, knowledge_base: Dict, num_queries: int = 1000) -> Dict:
        """创建合成查询（用于演示，实际应用应使用真实查询）"""
        items = knowledge_base.get('items', [])

        queries = []
        documents = []
        query_to_doc_labels = {}  # 查询索引 -> 相关文档索引列表

        # 文档文本
        for i, item in enumerate(items):
            # 创建文档文本
            doc_text = f"{item.get('category', '')} {item.get('title', '')} {item.get('scientific_basis', '')}"
            if 'warning_indicators' in item:
                doc_text += f" {item['warning_indicators']}"
            documents.append(doc_text)

        # 生成查询（基于文档内容）
        for query_idx in range(num_queries):
            # 随机选择一个文档作为查询基础
            doc_idx = np.random.randint(0, len(items))
            item = items[doc_idx]

            # 创建查询（模拟用户问题）
            if 'temperature' in item.get('scientific_basis', '').lower():
                query = f"温度{np.random.randint(30, 40)}℃ 湿度{np.random.randint(30, 80)}% 天气情况"
            elif 'rain' in item.get('scientific_basis', '').lower():
                query = f"降雨{np.random.randint(10, 100)}mm 预测"
            elif 'wind' in item.get('scientific_basis', '').lower():
                query = f"风速{np.random.randint(5, 20)}m/s 影响"
            else:
                query = f"{item.get('category', '')}相关咨询"

            queries.append(query)

            # 相关文档（选择同类别的文档）
            category = item.get('category', '')
            relevant_docs = []

            for j, other_item in enumerate(items):
                if other_item.get('category', '') == category:
                    relevant_docs.append(j)

            # 确保至少有一个相关文档
            if not relevant_docs:
                relevant_docs = [doc_idx]

            query_to_doc_labels[query_idx] = relevant_docs

        return {
            'queries': queries,
            'documents': documents,
            'labels': [query_to_doc_labels[i] for i in range(len(queries))]
        }

    def create_datasets(self, split_ratio: Tuple = (0.7, 0.15, 0.15)) -> Dict[str, KnowledgeDataset]:
        """创建数据集"""
        # 加载知识库
        knowledge_data = self.load_knowledge_base()
        knowledge_base = knowledge_data['knowledge_base']

        # 创建合成查询数据
        synthetic_data = self.create_synthetic_queries(knowledge_base, num_queries=2000)

        n_queries = len(synthetic_data['queries'])
        indices = np.random.permutation(n_queries)

        train_size = int(n_queries * split_ratio[0])
        val_size = int(n_queries * split_ratio[1])

        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]

        datasets = {}

        for name, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
            queries = [synthetic_data['queries'][i] for i in idx]
            labels = [synthetic_data['labels'][i] for i in idx]

            datasets[name] = KnowledgeDataset(
                queries=queries,
                documents=synthetic_data['documents'],
                labels=labels
            )

        print(f"数据集划分: 训练集={len(datasets['train'])}, "
              f"验证集={len(datasets['val'])}, 测试集={len(datasets['test'])}")

        return datasets

    def build_model(self, model_name: str = None):
        """构建模型"""
        if model_name is None:
            model_name = self.config.knowledge_config.get('base_model',
                                                          'paraphrase-multilingual-MiniLM-L12-v2')

        print(f"构建检索模型: {model_name}")

        # 加载预训练模型
        self.model = SentenceTransformer(model_name)

        print(f"模型维度: {self.model.get_sentence_embedding_dimension()}")

        return self.model

    def create_evaluator(self, val_dataset: KnowledgeDataset):
        """创建评估器"""
        # 准备评估数据
        # 将查询列表转换为字典 {query_id: query_text}
        queries = {str(i): query for i, query in enumerate(val_dataset.queries)}
        # 将文档列表转换为字典 {doc_id: doc_text}
        corpus = {str(i): doc for i, doc in enumerate(val_dataset.documents)}

        # 查询 -> 相关文档映射
        query_to_relevant_docs = {}
        for i, relevant_docs in enumerate(val_dataset.labels):
            query_to_relevant_docs[str(i)] = {str(doc_idx) for doc_idx in relevant_docs}

        # 创建评估器
        self.evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=query_to_relevant_docs,
            show_progress_bar=True,
            batch_size=32,
            name="climate_retrieval"
        )

    def train(self, train_dataset: KnowledgeDataset, val_dataset: KnowledgeDataset):
        """训练模型"""
        print("开始训练检索模型...")

        # 创建训练示例（使用sentence-transformers的InputExample格式）
        train_examples = train_dataset.create_triplet_examples()
        print(f"训练三元组数量: {len(train_examples)}")

        if len(train_examples) == 0:
            raise ValueError("没有训练样本，请检查数据")

        # 创建sentence-transformers的数据集
        train_data = SentencesDataset(train_examples, model=self.model)
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=self.config.training_config["batch_size"])

        # 创建损失函数
        train_loss = losses.TripletLoss(model=self.model)

        # 创建评估器
        self.create_evaluator(val_dataset)

        # 训练配置
        epochs = self.config.training_config["epochs"]
        warmup_steps = int(len(train_dataloader) * 0.1)

        # 训练模型 - 注意：移除了batch_size参数
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=self.evaluator,
            epochs=epochs,
            warmup_steps=warmup_steps,
            optimizer_params={'lr': self.config.training_config["learning_rate"]},
            output_path=str(self.output_dir / "model"),
            save_best_model=True,
            show_progress_bar=True,
            evaluation_steps=100,
            checkpoint_path=str(self.output_dir / "checkpoints"),
            checkpoint_save_steps=500
        )

        # 加载最佳模型
        self.model = SentenceTransformer(str(self.output_dir / "model"))

        # 记录历史
        self._record_training_history()

        self.logger.info("检索模型训练完成")

        return self.history

    def _record_training_history(self):
        """记录训练历史"""
        # 这里需要根据实际训练过程记录
        # 由于sentence-transformers的训练历史记录方式不同，这里简化处理
        pass

    def test(self, test_dataset: KnowledgeDataset) -> Dict:
        """测试模型"""
        print("测试检索模型...")

        # 编码所有文档
        corpus_embeddings = self.model.encode(
            test_dataset.documents,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # 测试指标
        results = {
            'precision_at_1': [],
            'precision_at_3': [],
            'precision_at_5': [],
            'recall_at_5': [],
            'ndcg_at_5': [],
            'mrr': []
        }

        # 对每个查询进行评估
        for query_idx, query in enumerate(test_dataset.queries):
            # 编码查询
            query_embedding = self.model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )[0]

            # 计算相似度
            similarities = np.dot(corpus_embeddings, query_embedding)

            # 获取排名
            ranked_indices = np.argsort(similarities)[::-1]

            # 相关文档
            relevant_docs = set(test_dataset.labels[query_idx])

            # 计算指标
            for k in [1, 3, 5]:
                retrieved_at_k = ranked_indices[:k]
                relevant_retrieved = len([idx for idx in retrieved_at_k if idx in relevant_docs])
                precision = relevant_retrieved / k if k > 0 else 0
                results[f'precision_at_{k}'].append(precision)

            # Recall@5
            relevant_retrieved = len([idx for idx in ranked_indices[:5] if idx in relevant_docs])
            recall = relevant_retrieved / len(relevant_docs) if relevant_docs else 0
            results['recall_at_5'].append(recall)

            # NDCG@5
            dcg = 0
            for rank, idx in enumerate(ranked_indices[:5], 1):
                if idx in relevant_docs:
                    dcg += 1 / np.log2(rank + 1)

            # 理想DCG
            ideal_ranking = min(5, len(relevant_docs))
            idcg = sum(1 / np.log2(i + 1) for i in range(1, ideal_ranking + 1))
            ndcg = dcg / idcg if idcg > 0 else 0
            results['ndcg_at_5'].append(ndcg)

            # MRR
            for rank, idx in enumerate(ranked_indices, 1):
                if idx in relevant_docs:
                    results['mrr'].append(1.0 / rank)
                    break
            else:
                results['mrr'].append(0.0)

        # 计算平均指标
        avg_results = {}
        for key, values in results.items():
            avg_results[key] = np.mean(values) if values else 0

        # 保存结果
        test_path = self.output_dir / "test_results.json"
        with open(test_path, 'w') as f:
            json.dump(avg_results, f, indent=2)

        print("测试结果:")
        for key, value in avg_results.items():
            print(f"  {key}: {value:.4f}")

        return avg_results

    def save_model(self):
        """保存模型"""
        model_path = self.output_dir / "adapted_model"
        self.model.save(str(model_path))

        # 保存配置
        config_path = model_path / "config.json"
        config_dict = {
            'model_name': self.model.model_name if hasattr(self.model, 'model_name') else str(self.model),
            'embedding_dim': self.model.get_sentence_embedding_dimension(),
            'training_config': self.config.training_config,
            'knowledge_config': self.config.knowledge_config
        }

        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        self.logger.info(f"模型保存到: {model_path}")

    def run_pipeline(self):
        """运行完整微调流水线"""
        self.logger.info("=" * 60)
        self.logger.info("📚 知识检索微调流水线")
        self.logger.info("=" * 60)

        try:
            # 1. 准备数据
            self.logger.info("步骤1: 准备数据...")
            datasets = self.create_datasets()

            if 'train' not in datasets or 'val' not in datasets:
                raise ValueError("缺少训练集或验证集")

            # 2. 构建模型
            self.logger.info("步骤2: 构建模型...")
            self.build_model()

            # 3. 训练模型
            self.logger.info("步骤3: 训练模型...")
            history = self.train(datasets['train'], datasets['val'])

            # 4. 测试模型
            self.logger.info("步骤4: 测试模型...")
            test_results = {}
            if 'test' in datasets:
                test_results = self.test(datasets['test'])

            # 5. 保存模型
            self.logger.info("步骤5: 保存模型...")
            self.save_model()

            self.logger.info("=" * 60)
            self.logger.info("✅ 知识检索微调完成!")
            self.logger.info("=" * 60)

            return {
                'model': self.model,
                'history': history,
                'test_results': test_results
            }

        except Exception as e:
            self.logger.error(f"微调失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise


def main():
    """主函数"""
    # 创建适应器
    adaptor = KnowledgeAdaptor()

    # 运行微调流水线
    results = adaptor.run_pipeline()

    return results


if __name__ == "__main__":
    main()