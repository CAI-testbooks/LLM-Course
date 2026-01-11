"""
气候文本知识处理模块 - 修复字体问题版本
处理JSON格式的气候知识文本，生成语义嵌入向量用于智能体系统
"""

import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import re
import hashlib
import sys

class ClimateKnowledgeProcessor:
    """气候知识文本处理器"""

    def __init__(self,
                 json_path: str = "/home/Liyang/agent/data/climate_knowledge.json",
                 output_dir: str = "/home/Liyang/agent/processed_data/text_features",
                 embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        初始化文本处理器
        """
        self.json_path = Path(json_path)
        self.output_dir = Path(output_dir)
        self.embedding_model_name = embedding_model

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 加载嵌入模型
        print(f"📚 加载文本嵌入模型: {self.embedding_model_name}")
        try:
            self.model = SentenceTransformer(self.embedding_model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"✓ 嵌入模型加载成功，维度: {self.embedding_dim}")
        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            # 使用备用模型
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"✓ 使用备用模型，维度: {self.embedding_dim}")

        # 数据存储
        self.knowledge_items = []
        self.embeddings = None
        self.metadata = {
            "total_items": 0,
            "categories": set(),
            "data_fields": set(),
            "processed_date": None
        }

    def load_and_validate_data(self) -> List[Dict]:
        """
        加载并验证JSON数据
        """
        print(f"\n📂 加载数据文件: {self.json_path}")

        if not self.json_path.exists():
            print(f"✗ 文件不存在: {self.json_path}")
            print("请创建示例文件或检查路径")
            return []

        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 验证数据格式
            if isinstance(data, list):
                self.knowledge_items = data
            elif isinstance(data, dict):
                # 如果是字典，可能是包含多个条目的对象
                if "knowledge" in data:
                    self.knowledge_items = data["knowledge"]
                elif "items" in data:
                    self.knowledge_items = data["items"]
                else:
                    # 尝试转换单个对象为列表
                    self.knowledge_items = [data]
            else:
                raise ValueError(f"❌ 未知的JSON格式: {type(data)}")

            print(f"✓ 成功加载 {len(self.knowledge_items)} 条知识条目")

            # 显示前几条数据
            for i, item in enumerate(self.knowledge_items[:3]):
                print(f"\n样本条目 {i+1}:")
                for key, value in item.items():
                    if isinstance(value, list):
                        print(f"  {key}: {', '.join(value[:5])}{'...' if len(value) > 5 else ''}")
                    else:
                        print(f"  {key}: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")

        except json.JSONDecodeError as e:
            print(f"❌ JSON解析错误: {e}")
            print(f"请检查 {self.json_path} 文件格式")
            return []
        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            return []

        # 验证每个条目的结构
        valid_items = []
        for i, item in enumerate(self.knowledge_items):
            if self._validate_item(item, i):
                valid_items.append(item)

        self.knowledge_items = valid_items
        print(f"✓ 有效条目: {len(self.knowledge_items)} 条")

        # 提取元数据
        self._extract_metadata()

        return self.knowledge_items

    def _validate_item(self, item: Dict, index: int) -> bool:
        """
        验证单个知识条目的结构
        """
        # 必需字段检查
        required_fields = ['category', 'title', 'scientific_basis']
        missing_fields = [field for field in required_fields if field not in item]

        if missing_fields:
            print(f"⚠️ 条目 {index} 缺少必需字段: {missing_fields}")
            return False

        # 字段类型检查
        if not isinstance(item['category'], str):
            print(f"⚠️ 条目 {index} 的category字段不是字符串类型")
            return False

        if not isinstance(item['title'], str):
            print(f"⚠️ 条目 {index} 的title字段不是字符串类型")
            return False

        if not isinstance(item['scientific_basis'], str):
            print(f"⚠️ 条目 {index} 的scientific_basis字段不是字符串类型")
            return False

        return True

    def _extract_metadata(self):
        """提取数据集的元数据"""
        self.metadata["total_items"] = len(self.knowledge_items)
        self.metadata["processed_date"] = pd.Timestamp.now().isoformat()

        # 收集所有类别
        categories = set()
        data_fields = set()

        for item in self.knowledge_items:
            categories.add(item['category'])

            # 收集相关数据字段
            if 'related_data_fields' in item and isinstance(item['related_data_fields'], list):
                for field in item['related_data_fields']:
                    data_fields.add(field)

        self.metadata["categories"] = list(categories)
        self.metadata["data_fields"] = list(data_fields)

    def preprocess_text(self, item: Dict) -> List[str]:
        """
        预处理单个知识条目，生成多个文本表示
        """
        texts = []

        # 1. 完整表示（所有信息）
        full_text = f"类别：{item['category']} 标题：{item['title']} 科学依据：{item['scientific_basis']}"

        if 'warning_indicators' in item:
            full_text += f" 预警指标：{item['warning_indicators']}"

        if 'adaptive_actions' in item:
            full_text += f" 应对措施：{item['adaptive_actions']}"

        texts.append(full_text.strip())

        # 2. 科学依据+预警指标（用于风险评估）
        if 'warning_indicators' in item:
            risk_text = f"{item['scientific_basis']} {item['warning_indicators']}"
            texts.append(risk_text)

        # 3. 简洁表示（用于快速检索）
        concise_text = f"{item['category']}：{item['title']} - {item['scientific_basis'][:100]}..."
        texts.append(concise_text)

        # 4. 仅科学依据（用于匹配气象数据）
        texts.append(item['scientific_basis'])

        return texts

    def extract_keywords(self, text: str) -> List[str]:
        """
        提取文本中的关键词
        """
        # 提取中文关键词
        chinese_words = re.findall(r'[\u4e00-\u9fff]{2,}', text)

        # 提取英文变量名（如 Present_Tmax）
        english_vars = re.findall(r'[A-Z][A-Za-z_]+', text)

        # 提取温度阈值（如 32℃, 35℃）
        temperature_thresholds = re.findall(r'\d+℃', text)

        # 提取百分比（如 45%）
        percentages = re.findall(r'\d+%', text)

        # 合并所有关键词
        keywords = chinese_words + english_vars + temperature_thresholds + percentages

        # 去重并过滤空字符串
        keywords = [kw for kw in set(keywords) if kw.strip()]

        return keywords

    def generate_embeddings(self, batch_size: int = 32) -> np.ndarray:
        """
        为所有知识条目生成嵌入向量
        """
        print(f"\n🔧 生成文本嵌入向量...")

        if not self.knowledge_items:
            print("❌ 没有可处理的知识条目")
            return np.array([])

        # 准备文本数据
        all_texts = []
        text_indices = []  # 记录每个条目对应的文本索引

        for idx, item in enumerate(tqdm(self.knowledge_items, desc="准备文本")):
            texts = self.preprocess_text(item)
            all_texts.extend(texts)
            text_indices.append((idx, len(texts)))  # (条目索引, 文本数量)

        print(f"✓ 共生成 {len(all_texts)} 个文本表示")

        # 生成嵌入向量
        print(f"⏳ 正在计算嵌入向量...")
        embeddings = self.model.encode(
            all_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # 归一化以便后续计算余弦相似度
        )

        print(f"✓ 嵌入向量生成完成，形状: {embeddings.shape}")

        # 重新组织嵌入向量：为每个条目生成一个综合嵌入
        self.embeddings = np.zeros((len(self.knowledge_items), self.embedding_dim))

        current_idx = 0
        for item_idx, (original_idx, num_texts) in enumerate(text_indices):
            # 获取该条目的所有文本嵌入
            item_embeddings = embeddings[current_idx:current_idx + num_texts]

            # 使用平均池化生成综合嵌入
            self.embeddings[item_idx] = np.mean(item_embeddings, axis=0)

            current_idx += num_texts

        # 归一化最终嵌入
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # 避免除以零
        self.embeddings = self.embeddings / norms

        return self.embeddings

    def create_knowledge_base(self) -> Dict:
        """
        创建结构化的知识库
        """
        print(f"\n🏗️ 创建结构化知识库...")

        knowledge_base = {
            "metadata": self.metadata,
            "items": [],
            "embeddings": self.embeddings.tolist() if self.embeddings is not None else [],
            "index_mapping": {}  # 类别到条目索引的映射
        }

        # 按类别组织条目
        category_to_indices = {}

        for idx, item in enumerate(self.knowledge_items):
            # 复制条目并添加处理后的信息
            processed_item = item.copy()

            # 提取关键词
            full_text = self.preprocess_text(item)[0]
            keywords = self.extract_keywords(full_text)
            processed_item['keywords'] = keywords

            # 生成条目ID
            item_hash = hashlib.md5(full_text.encode()).hexdigest()[:8]
            processed_item['item_id'] = f"KNOW_{item_hash}"

            # 添加到知识库
            knowledge_base["items"].append(processed_item)

            # 更新类别映射
            category = item['category']
            if category not in category_to_indices:
                category_to_indices[category] = []
            category_to_indices[category].append(idx)

        # 更新索引映射
        knowledge_base["index_mapping"] = category_to_indices

        print(f"✓ 知识库创建完成，包含 {len(knowledge_base['items'])} 个条目")

        # 显示类别分布
        print(f"✓ 类别分布:")
        for cat, indices in category_to_indices.items():
            print(f"  {cat}: {len(indices)} 条")

        return knowledge_base

    def save_results(self, knowledge_base: Dict):
        """
        保存处理结果
        """
        print(f"\n💾 保存处理结果到 {self.output_dir}")

        # 1. 保存完整知识库为JSON
        json_path = self.output_dir / "knowledge_base.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(knowledge_base, f, ensure_ascii=False, indent=2)
        print(f"✓ 知识库保存到: {json_path}")

        # 2. 保存嵌入向量为numpy文件
        if self.embeddings is not None:
            np_path = self.output_dir / "embeddings.npy"
            np.save(np_path, self.embeddings)
            print(f"✓ 嵌入向量保存到: {np_path}")

        # 3. 保存元数据
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        print(f"✓ 元数据保存到: {metadata_path}")

        # 4. 保存为CSV格式（便于查看）
        csv_data = []
        for item in knowledge_base["items"]:
            csv_row = {
                "item_id": item.get("item_id", ""),
                "category": item.get("category", ""),
                "title": item.get("title", ""),
                "scientific_basis": item.get("scientific_basis", ""),
                "keywords": ", ".join(item.get("keywords", [])),
                "related_data_fields": ", ".join(item.get("related_data_fields", [])),
                "warning_indicators": item.get("warning_indicators", ""),
                "adaptive_actions": item.get("adaptive_actions", "")
            }
            csv_data.append(csv_row)

        csv_path = self.output_dir / "knowledge_base.csv"
        pd.DataFrame(csv_data).to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✓ CSV格式保存到: {csv_path}")

        # 5. 保存为pickle格式（用于Python快速加载）
        pkl_path = self.output_dir / "knowledge_base.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(knowledge_base, f)
        print(f"✓ Pickle格式保存到: {pkl_path}")

    def analyze_knowledge_base(self, knowledge_base: Dict):
        """
        分析知识库并生成统计报告
        """
        print(f"\n📊 知识库分析报告")
        print("=" * 50)

        items = knowledge_base["items"]

        # 1. 基本统计
        print(f"知识条目总数: {len(items)}")
        print(f"类别数量: {len(knowledge_base['index_mapping'])}")

        # 2. 类别分布
        print("\n📈 类别分布:")
        for category, indices in knowledge_base["index_mapping"].items():
            percentage = len(indices) / len(items) * 100
            print(f"  {category}: {len(indices)} 条 ({percentage:.1f}%)")

        # 3. 关键词统计
        all_keywords = []
        for item in items:
            all_keywords.extend(item.get("keywords", []))

        from collections import Counter
        keyword_counts = Counter(all_keywords)

        print(f"\n🔑 高频关键词 (Top 15):")
        for keyword, count in keyword_counts.most_common(15):
            print(f"  {keyword}: {count} 次")

        # 4. 数据字段统计
        all_data_fields = []
        for item in items:
            all_data_fields.extend(item.get("related_data_fields", []))

        data_field_counts = Counter(all_data_fields)

        print(f"\n📋 相关数据字段统计:")
        for field, count in data_field_counts.most_common():
            print(f"  {field}: {count} 次")

        # 5. 文本长度分析
        text_lengths = []
        for item in items:
            full_text = self.preprocess_text(item)[0]
            text_lengths.append(len(full_text))

        print(f"\n📏 文本长度统计:")
        print(f"  平均长度: {np.mean(text_lengths):.0f} 字符")
        print(f"  最小长度: {np.min(text_lengths)} 字符")
        print(f"  最大长度: {np.max(text_lengths)} 字符")
        print(f"  标准差: {np.std(text_lengths):.0f} 字符")

    def visualize_knowledge_base_simple(self, knowledge_base: Dict):
        """
        简化版可视化（不需要中文字体）
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            print(f"\n🎨 生成简化可视化图表...")

            # 创建可视化目录
            viz_dir = self.output_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)

            # 1. 类别分布柱状图（使用英文标签避免字体问题）
            categories = list(knowledge_base["index_mapping"].keys())
            counts = [len(indices) for indices in knowledge_base["index_mapping"].values()]

            # 生成简短的类别标签
            short_categories = []
            for cat in categories:
                # 取前几个字符或使用缩写
                if len(cat) > 10:
                    short_cat = cat[:8] + "..."
                else:
                    short_cat = cat
                short_categories.append(short_cat)

            plt.figure(figsize=(12, 6))
            bars = plt.bar(range(len(categories)), counts, alpha=0.7)
            plt.xticks(range(len(categories)), short_categories, rotation=45, ha='right')
            plt.title('Knowledge Category Distribution')
            plt.xlabel('Category')
            plt.ylabel('Count')

            # 在柱子上添加数量标签
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{count}', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(viz_dir / "category_distribution_en.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ 类别分布图保存到: {viz_dir}/category_distribution_en.png")

            # 2. 关键词频率柱状图
            all_keywords = []
            for item in knowledge_base["items"]:
                all_keywords.extend(item.get("keywords", []))

            from collections import Counter
            keyword_counts = Counter(all_keywords)

            # 取前20个关键词
            top_keywords = keyword_counts.most_common(15)
            keywords, counts = zip(*top_keywords) if top_keywords else ([], [])

            plt.figure(figsize=(12, 6))
            bars = plt.bar(range(len(keywords)), counts, alpha=0.7)
            plt.xticks(range(len(keywords)), keywords, rotation=45, ha='right')
            plt.title('Top Keywords Frequency')
            plt.xlabel('Keyword')
            plt.ylabel('Frequency')

            plt.tight_layout()
            plt.savefig(viz_dir / "keyword_frequency_en.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ 关键词频率图保存到: {viz_dir}/keyword_frequency_en.png")

        except ImportError as e:
            print(f"⚠️ 可视化依赖库未安装: {e}")
            print("请运行: pip install matplotlib seaborn")

    def visualize_knowledge_base_advanced(self, knowledge_base: Dict):
        """
        高级可视化（尝试使用中文字体）
        """
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.manifold import TSNE

            print(f"\n🎨 生成高级可视化图表...")

            # 创建可视化目录
            viz_dir = self.output_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)

            # 尝试设置中文字体（如果可用）
            def try_set_chinese_font():
                """尝试设置中文字体"""
                # 常见的中文字体路径
                font_paths = [
                    # Windows
                    "C:/Windows/Fonts/simhei.ttf",  # 黑体
                    "C:/Windows/Fonts/simsun.ttc",  # 宋体
                    # Mac
                    "/System/Library/Fonts/PingFang.ttc",
                    "/System/Library/Fonts/STHeiti Medium.ttc",
                    # Linux
                    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                    "/usr/share/fonts/truetype/arphic/uming.ttc",
                ]

                for font_path in font_paths:
                    if Path(font_path).exists():
                        try:
                            matplotlib.font_manager.fontManager.addfont(font_path)
                            font_name = matplotlib.font_manager.FontProperties(fname=font_path).get_name()
                            matplotlib.rcParams['font.sans-serif'] = [font_name]
                            matplotlib.rcParams['axes.unicode_minus'] = False
                            print(f"✓ 使用中文字体: {font_name}")
                            return True
                        except:
                            continue

                print("⚠️ 未找到可用的中文字体，使用默认字体")
                return False

            # 尝试设置中文字体
            has_chinese_font = try_set_chinese_font()

            # 1. 类别分布饼图
            categories = list(knowledge_base["index_mapping"].keys())
            counts = [len(indices) for indices in knowledge_base["index_mapping"].values()]

            plt.figure(figsize=(10, 8))
            plt.pie(counts, labels=categories, autopct='%1.1f%%', startangle=90)
            plt.title('知识条目类别分布')
            plt.tight_layout()

            if has_chinese_font:
                plt.savefig(viz_dir / "category_distribution_cn.png", dpi=300, bbox_inches='tight')
                print(f"✓ 中文类别分布图保存到: {viz_dir}/category_distribution_cn.png")
            else:
                # 如果无法显示中文，使用英文标签
                plt.title('Knowledge Category Distribution')
                plt.savefig(viz_dir / "category_distribution.png", dpi=300, bbox_inches='tight')
                print(f"✓ 类别分布图保存到: {viz_dir}/category_distribution.png")

            plt.close()

            # 2. 嵌入向量可视化（t-SNE）
            if self.embeddings is not None and len(self.embeddings) > 10:
                # 降维
                n_samples = min(100, len(self.embeddings))
                indices = np.random.choice(len(self.embeddings), n_samples, replace=False)
                embeddings_sample = self.embeddings[indices]

                # t-SNE
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples-1))
                embeddings_2d = tsne.fit_transform(embeddings_sample)

                # 获取类别标签
                labels = []
                for idx in indices:
                    # 找到对应的条目
                    for category, cat_indices in knowledge_base["index_mapping"].items():
                        if idx in cat_indices:
                            labels.append(category)
                            break
                    else:
                        labels.append("Unknown")

                # 绘制
                plt.figure(figsize=(12, 10))
                scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                                    c=[hash(label) % 20 for label in labels],
                                    cmap='tab20', alpha=0.7, s=100)

                # 添加图例
                unique_labels = list(set(labels))
                handles = [plt.Line2D([0], [0], marker='o', color='w',
                                     markerfacecolor=plt.cm.tab20(hash(label) % 20 / 20),
                                     markersize=10) for label in unique_labels]
                plt.legend(handles, unique_labels, title="类别", bbox_to_anchor=(1.05, 1), loc='upper left')

                plt.title('知识条目嵌入向量可视化 (t-SNE)')
                plt.xlabel('t-SNE 维度 1')
                plt.ylabel('t-SNE 维度 2')
                plt.tight_layout()
                plt.savefig(viz_dir / "embeddings_tsne.png", dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✓ 嵌入向量可视化保存到: {viz_dir}/embeddings_tsne.png")

        except ImportError as e:
            print(f"⚠️ 高级可视化依赖库未安装: {e}")
            print("请运行: pip install matplotlib seaborn scikit-learn")

    def run_pipeline(self, enable_visualization: bool = True):
        """
        运行完整的数据处理流水线
        """
        print("=" * 60)
        print("🌍 气候文本知识处理流水线")
        print("=" * 60)

        # 步骤1: 加载数据
        data = self.load_and_validate_data()

        if not data:
            print("❌ 没有数据可处理，程序退出")
            return None

        # 步骤2: 生成嵌入向量
        self.generate_embeddings()

        # 步骤3: 创建知识库
        knowledge_base = self.create_knowledge_base()

        # 步骤4: 分析知识库
        self.analyze_knowledge_base(knowledge_base)

        # 步骤5: 可视化（可选）
        if enable_visualization:
            try:
                # 尝试高级可视化
                self.visualize_knowledge_base_advanced(knowledge_base)
            except Exception as e:
                print(f"⚠️ 高级可视化失败: {e}")
                # 回退到简化版
                self.visualize_knowledge_base_simple(knowledge_base)

        # 步骤6: 保存结果
        self.save_results(knowledge_base)

        print("\n" + "=" * 60)
        print("✅ 气候文本数据处理完成!")
        print("=" * 60)

        # 打印输出文件摘要
        print(f"\n📁 输出文件:")
        for file_path in self.output_dir.iterdir():
            if file_path.is_file():
                file_size = file_path.stat().st_size / 1024  # KB
                print(f"  {file_path.name} ({file_size:.1f} KB)")

        return knowledge_base

# ============================ 命令行接口 ============================
def process_climate_knowledge():
    """处理气候知识文本的主函数"""

    # 创建处理器实例
    processor = ClimateKnowledgeProcessor(
        json_path="/home/Liyang/agent/knowledge_base/knowledge_base.json",
        output_dir="/home/Liyang/agent/knowledge_base",
        embedding_model="paraphrase-multilingual-MiniLM-L12-v2"
    )

    # 运行完整流水线
    knowledge_base = processor.run_pipeline(enable_visualization=True)

    if knowledge_base:
        print(f"\n🔗 智能体系统集成示例:")
        print("""
    在您的智能体系统中使用：
    
    1. 加载知识库：
        import pickle
        with open('/home/Liyang/agent/processed_data/text_features/knowledge_base.pkl', 'rb') as f:
            knowledge_base = pickle.load(f)
    
    2. 检索相关知识：
        import numpy as np
        from sentence_transformers import SentenceTransformer
        
        # 加载模型
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # 定义检索函数
        def retrieve_relevant_knowledge(query_text, knowledge_base, top_k=3):
            # 生成查询嵌入
            query_embedding = model.encode([query_text])[0]
            
            # 计算相似度
            similarities = np.dot(knowledge_base['embeddings'], query_embedding)
            
            # 返回最相关的知识
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            results = []
            for idx in top_indices:
                item = knowledge_base['items'][idx]
                results.append({
                    'item_id': item['item_id'],
                    'category': item['category'],
                    'title': item['title'],
                    'scientific_basis': item['scientific_basis'],
                    'similarity': similarities[idx]
                })
            return results
    
    3. 示例查询：
        # 假设有气象数据
        weather_data = {
            'Present_Tmax': 35.5,
            'LDAPS_RHmin': 40.0
        }
        
        # 创建查询文本
        query = f"温度{weather_data['Present_Tmax']}℃ 湿度{weather_data['LDAPS_RHmin']}%"
        
        # 检索相关知识
        relevant_knowledge = retrieve_relevant_knowledge(query, knowledge_base)
        
        for i, knowledge in enumerate(relevant_knowledge):
            print(f"{i+1}. [{knowledge['category']}] {knowledge['title']}")
            print(f"   相似度: {knowledge['similarity']:.3f}")
            print(f"   科学依据: {knowledge['scientific_basis'][:100]}...")
            print()
        """)

    return knowledge_base

# ============================ 安装依赖脚本 ============================
def install_dependencies():
    """安装必要的依赖包"""
    print("安装文本处理所需依赖...")

    dependencies = [
        "sentence-transformers",
        "numpy",
        "pandas",
        "tqdm",
        "scikit-learn",
        "matplotlib",
        "seaborn",
    ]

    import subprocess
    import sys

    for package in dependencies:
        print(f"正在安装 {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} 安装成功")
        except subprocess.CalledProcessError:
            print(f"✗ {package} 安装失败，请手动安装: pip install {package}")

    print("\n所有依赖安装完成！")

# ============================ 创建示例数据脚本 ============================
def create_sample_data():
    """创建示例JSON数据文件"""
    sample_data = [
        {
            "category": "户外赛事热浪安全",
            "title": "户外赛事 / 活动的热浪安全阈值",
            "scientific_basis": "Present_Tmax≥32℃+LDAPS_RHmin≤45% 时，长时间户外剧烈运动易引发群体性热射病，需设定赛事举办的安全红线",
            "related_data_fields": ["Present_Tmax", "LDAPS_RHmin", "lat", "lon"],
            "warning_indicators": "低风险：Present_Tmax<32℃；中风险：32℃≤Present_Tmax<34℃；高风险：Present_Tmax≥34℃或 LDAPS_RHmin≤45%",
            "adaptive_actions": "低风险时正常举办；中风险时缩短赛事时长、增加补水点；高风险时延期或取消赛事"
        },
        {
            "category": "农业热浪预警",
            "title": "农作物高温热害预警",
            "scientific_basis": "连续3天Present_Tmax≥35℃或日最高气温≥38℃时，水稻、玉米等作物易受高温热害，影响授粉灌浆",
            "related_data_fields": ["Present_Tmax", "Present_Tmin", "LDAPS_RHmin"],
            "warning_indicators": "轻度热害：35℃≤Present_Tmax<37℃持续3天；中度热害：Present_Tmax≥37℃持续2天；重度热害：Present_Tmax≥40℃",
            "adaptive_actions": "轻度时增加灌溉；中度时喷施叶面肥；重度时考虑提前收割"
        },
        {
            "category": "城市热岛效应",
            "title": "城市热浪健康风险",
            "scientific_basis": "城市地区由于热岛效应，夜间温度比郊区高2-5℃，增加居民热相关疾病风险，特别是老年人和儿童",
            "related_data_fields": ["Present_Tmin", "LDAPS_Tmax_lapse", "lat", "lon", "DEM"],
            "warning_indicators": "关注夜间温度Present_Tmin≥28℃或日温差<5℃的情况",
            "adaptive_actions": "开放避暑场所，延长公共场所开放时间，发布健康提醒"
        },
        {
            "category": "能源电力需求",
            "title": "高温天气电力负荷预测",
            "scientific_basis": "气温每升高1℃，城市电力负荷增加3-5%，当Present_Tmax≥35℃时，空调负荷可能占总负荷的40%以上",
            "related_data_fields": ["Present_Tmax", "LDAPS_Tmax_lapse", "Solar radiation"],
            "warning_indicators": "黄色预警：35℃≤Present_Tmax<37℃；橙色预警：37℃≤Present_Tmax<39℃；红色预警：Present_Tmax≥39℃",
            "adaptive_actions": "黄色预警时启动有序用电预案；橙色预警时限制工业用电；红色预警时采取轮换停电措施"
        },
        {
            "category": "交通出行安全",
            "title": "高温天气道路安全",
            "scientific_basis": "路面温度可达气温的1.5-2倍，当Present_Tmax≥35℃时，沥青路面温度可能超过60℃，增加爆胎风险",
            "related_data_fields": ["Present_Tmax", "Solar radiation", "LDAPS_WS"],
            "warning_indicators": "道路高温预警：路面温度≥55℃；交通管制建议：路面温度≥65℃",
            "adaptive_actions": "高温时段减少运输任务，增加道路洒水降温，提醒车辆检查胎压"
        }
    ]

    output_path = Path("/home/Liyang/agent/knowledge_base")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)

    print(f"✓ 示例数据已创建: {output_path}")
    print(f"✓ 包含 {len(sample_data)} 条知识条目")

    return sample_data

# ============================ 主执行函数 ============================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='气候文本知识处理工具')
    parser.add_argument('--install', action='store_true', help='安装依赖包')
    parser.add_argument('--create-sample', action='store_true', help='创建示例数据')
    parser.add_argument('--process', action='store_true', help='处理文本数据')
    parser.add_argument('--no-visualization', action='store_true', help='不生成可视化图表')

    args = parser.parse_args()

    if args.install:
        install_dependencies()

    if args.create_sample:
        create_sample_data()

    if args.process or (not args.install and not args.create_sample):
        # 默认运行数据处理
        process_climate_knowledge()

    if not any([args.install, args.create_sample, args.process]):
        # 如果没有指定任何参数，显示帮助信息
        print("气候文本知识处理工具")
        print("=" * 50)
        print("使用方法:")
        print("  python climate_text_processor.py --install      # 安装依赖")
        print("  python climate_text_processor.py --create-sample # 创建示例数据")
        print("  python climate_text_processor.py --process      # 处理文本数据")
        print("\n或直接运行: python climate_text_processor.py")
