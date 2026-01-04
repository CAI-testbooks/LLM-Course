"""
气象智能RAG系统 - 简化版本
修复transformers版本兼容性问题
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import json
import pickle
from pathlib import Path
import warnings
import logging
import re
from datetime import datetime
import sys

warnings.filterwarnings('ignore')

from sentence_transformers import SentenceTransformer, util
from config import config


class KnowledgeBaseRetriever:
    """知识库检索器"""

    def __init__(self, knowledge_path: str = None, model_name: str = None):
        """
        初始化知识库检索器

        Args:
            knowledge_path: 知识库文件路径
            model_name: 检索模型名称
        """
        self.knowledge_path = knowledge_path or config.paths.get("knowledge_json", "/home/Liyang/agent/knowledge_base.json")
        self.model_name = model_name or config.knowledge_config.get('base_model', 'paraphrase-multilingual-MiniLM-L12-v2')

        # 加载知识库
        self.knowledge_base = self.load_knowledge_base()

        # 初始化检索模型
        self.retrieval_model = SentenceTransformer(self.model_name)

        # 构建文档和向量
        self.documents, self.document_embeddings = self.prepare_documents()

        print(f"知识库检索器初始化完成: {len(self.documents)} 文档")

    def load_knowledge_base(self) -> Dict:
        """加载知识库"""
        knowledge_path = Path(self.knowledge_path)

        if not knowledge_path.exists():
            print(f"警告: 知识库文件不存在: {knowledge_path}")
            return {"items": []}

        try:
            with open(knowledge_path, 'r', encoding='utf-8') as f:
                knowledge_base = json.load(f)
            print(f"成功加载知识库: {len(knowledge_base.get('items', []))} 个条目")
            return knowledge_base
        except Exception as e:
            print(f"加载知识库失败: {e}")
            return {"items": []}

    def format_document(self, item: Dict) -> str:
        """格式化知识库条目为文档文本"""
        parts = []

        # 标题
        if 'title' in item:
            parts.append(f"标题: {item['title']}")

        # 类别
        if 'category' in item:
            parts.append(f"类别: {item['category']}")

        # 科学依据
        if 'scientific_basis' in item:
            parts.append(f"科学依据: {item['scientific_basis']}")

        # 预警指标
        if 'warning_indicators' in item:
            parts.append(f"预警指标: {item['warning_indicators']}")

        # 影响与应对
        if 'impact_response' in item:
            parts.append(f"影响与应对: {item['impact_response']}")

        return "\n".join(parts)

    def prepare_documents(self) -> Tuple[List[str], np.ndarray]:
        """准备文档和嵌入向量"""
        documents = []
        items = self.knowledge_base.get('items', [])

        for i, item in enumerate(items):
            doc_text = self.format_document(item)
            documents.append(doc_text)

        if not documents:
            # 创建一些示例文档
            documents = [
                "高温天气容易导致中暑，建议减少户外活动，多喝水，避免中午时段外出。",
                "暴雨天气可能导致城市内涝，请注意交通安全，避免涉水行车。",
                "干旱天气需要节约用水，注意防火，减少户外活动。",
                "台风天气风力强劲，请固定好门窗，避免外出。",
                "寒潮天气气温骤降，请注意保暖，预防感冒。"
            ]
            print("使用示例文档，因为没有加载到实际知识库")

        # 计算文档嵌入向量
        print(f"计算文档嵌入向量...")
        document_embeddings = self.retrieval_model.encode(
            documents,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return documents, document_embeddings

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """检索相关文档"""
        if len(self.documents) == 0:
            return []

        # 编码查询
        query_embedding = self.retrieval_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # 计算相似度
        similarities = util.cos_sim(query_embedding, self.document_embeddings)[0]

        # 获取top_k结果
        top_indices = torch.topk(similarities, k=min(top_k, len(self.documents))).indices.tolist()

        # 构建结果
        results = []
        for idx in top_indices:
            similarity = similarities[idx].item()
            results.append({
                'document': self.documents[idx],
                'similarity': similarity,
                'rank': len(results) + 1
            })

        return results


class WeatherFeatureExtractor:
    """天气特征提取器"""

    def __init__(self):
        self.features = {}

    def extract(self, query: str) -> Dict:
        """从查询中提取天气特征"""
        features = {}

        # 温度提取
        temp_patterns = [
            r'温度\s*([0-9]+\.?[0-9]*)\s*℃',
            r'([0-9]+\.?[0-9]*)\s*℃',
            r'气温\s*([0-9]+\.?[0-9]*)'
        ]

        for pattern in temp_patterns:
            match = re.search(pattern, query)
            if match:
                try:
                    temp = float(match.group(1))
                    features['temperature'] = temp

                    if temp >= 35:
                        features['heat_level'] = '酷热'
                    elif temp >= 30:
                        features['heat_level'] = '炎热'
                    elif temp >= 25:
                        features['heat_level'] = '温暖'
                    elif temp >= 15:
                        features['heat_level'] = '舒适'
                    else:
                        features['heat_level'] = '寒冷'

                    break
                except ValueError:
                    continue

        # 湿度提取
        humidity_patterns = [
            r'湿度\s*([0-9]+\.?[0-9]*)\s*%',
            r'([0-9]+\.?[0-9]*)\s*%湿度'
        ]

        for pattern in humidity_patterns:
            match = re.search(pattern, query)
            if match:
                try:
                    humidity = float(match.group(1))
                    features['humidity'] = humidity

                    if humidity >= 80:
                        features['humidity_level'] = '高湿'
                    elif humidity >= 60:
                        features['humidity_level'] = '中等'
                    elif humidity >= 40:
                        features['humidity_level'] = '舒适'
                    else:
                        features['humidity_level'] = '干燥'

                    break
                except ValueError:
                    continue

        # 天气现象关键词
        weather_keywords = {
            '降雨': 'rain',
            '降水': 'precipitation',
            '下雨': 'rain',
            '暴雨': 'heavy_rain',
            '大风': 'wind',
            '台风': 'typhoon',
            '热浪': 'heatwave',
            '高温': 'high_temperature',
            '干旱': 'drought',
            '寒潮': 'cold_wave',
            '霜冻': 'frost',
            '雾霾': 'haze',
            '沙尘': 'sandstorm'
        }

        for keyword, feature_key in weather_keywords.items():
            if keyword in query:
                features[feature_key] = True

        # 时间关键词
        time_keywords = {
            '今天': 'today',
            '明天': 'tomorrow',
            '后天': 'day_after_tomorrow',
            '本周': 'this_week',
            '周末': 'weekend',
            '未来三天': 'next_three_days',
            '下周': 'next_week'
        }

        for keyword, time_key in time_keywords.items():
            if keyword in query:
                features['time_period'] = time_key
                break

        return features


class ResponseGenerator:
    """响应生成器"""

    def __init__(self):
        # 响应模板
        self.templates = {
            'high_temperature': [
                "根据查询，当前天气温度较高，需要注意防暑降温。",
                "高温天气容易引发中暑，建议减少户外活动。",
                "请做好防晒措施，避免在高温时段进行剧烈运动。"
            ],
            'rain': [
                "查询涉及降雨天气，请注意携带雨具。",
                "降雨可能影响出行，建议提前规划路线。",
                "雨天路滑，请注意交通安全。"
            ],
            'drought': [
                "干旱天气需要特别注意节约用水。",
                "高温干旱天气容易引发火灾，请注意防火。",
                "建议减少户外活动，避免长时间暴露在干燥环境中。"
            ],
            'wind': [
                "大风天气请注意安全，避免在广告牌、临时搭建物下停留。",
                "建议固定好门窗和室外物品，防止被风吹落。"
            ],
            'general': [
                "根据知识库检索结果，为您提供以下信息：",
                "结合气象特征分析，建议您：",
                "综合来看，需要注意以下几点："
            ]
        }

    def generate(self, query: str, retrieved_docs: List[Dict],
                 weather_features: Dict) -> Dict:
        """生成响应"""

        # 分析天气特征
        analysis_parts = []

        if 'temperature' in weather_features:
            temp = weather_features['temperature']
            if temp >= 35:
                analysis_parts.append(f"温度高达{temp}℃，属于酷热天气")
            elif temp >= 30:
                analysis_parts.append(f"温度{temp}℃，属于炎热天气")
            elif temp >= 25:
                analysis_parts.append(f"温度{temp}℃，较为温暖")
            else:
                analysis_parts.append(f"温度{temp}℃，较为凉爽")

        if 'humidity' in weather_features:
            humidity = weather_features['humidity']
            if humidity >= 80:
                analysis_parts.append(f"湿度{humidity}%，较为潮湿")
            elif humidity <= 40:
                analysis_parts.append(f"湿度{humidity}%，较为干燥")
            else:
                analysis_parts.append(f"湿度{humidity}%，较为舒适")

        # 提取检索文档的关键信息
        doc_summaries = []
        for doc in retrieved_docs[:3]:  # 取前3个文档
            doc_text = doc['document']
            similarity = doc['similarity']

            # 提取摘要（取前100个字符）
            summary = doc_text[:100] + "..." if len(doc_text) > 100 else doc_text
            doc_summaries.append({
                'summary': summary,
                'similarity': similarity
            })

        # 生成建议
        recommendations = []

        if weather_features.get('high_temperature'):
            recommendations.extend([
                "避免在10:00-16:00高温时段进行户外活动",
                "穿戴宽松、透气的衣物，佩戴太阳镜和遮阳帽",
                "及时补充水分，不要等到口渴才喝水",
                "如出现头晕、恶心等中暑症状，立即到阴凉处休息"
            ])

        if weather_features.get('rain'):
            recommendations.extend([
                "出门前查看天气预报，携带雨具",
                "行车时注意减速慢行，保持安全车距",
                "避免在树下、电线杆下避雨",
                "注意防范雷电天气"
            ])

        if weather_features.get('drought'):
            recommendations.extend([
                "节约用水，减少不必要的用水",
                "注意防火，不要乱扔烟头",
                "避免在户外使用明火",
                "做好皮肤保湿，防止皮肤干燥"
            ])

        # 如果没有特定建议，添加通用建议
        if not recommendations:
            recommendations = [
                "关注当地气象部门的最新预报和预警",
                "根据天气变化及时调整出行计划",
                "做好个人防护，保持健康生活方式"
            ]

        # 构建响应
        response = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'weather_analysis': analysis_parts,
            'retrieved_docs_count': len(retrieved_docs),
            'doc_summaries': doc_summaries,
            'recommendations': recommendations,
            'confidence': min(0.9, retrieved_docs[0]['similarity'] if retrieved_docs else 0.5)
        }

        # 生成自然语言响应
        response['natural_response'] = self._generate_natural_response(response)

        return response

    def _generate_natural_response(self, response: Dict) -> str:
        """生成自然语言响应"""
        lines = []

        lines.append(f"📊 针对您的查询「{response['query']}」，分析如下：")
        lines.append("")

        # 天气分析
        if response['weather_analysis']:
            lines.append("🌤️ **天气特征分析**")
            for analysis in response['weather_analysis']:
                lines.append(f"• {analysis}")
            lines.append("")

        # 检索结果
        lines.append(f"🔍 **知识库检索结果**（共{response['retrieved_docs_count']}条相关文档）")
        for i, doc in enumerate(response['doc_summaries'][:3], 1):
            lines.append(f"{i}. {doc['summary']} (相关度: {doc['similarity']:.2f})")
        lines.append("")

        # 建议
        lines.append("💡 **建议措施**")
        for i, rec in enumerate(response['recommendations'][:5], 1):
            lines.append(f"{i}. {rec}")

        lines.append("")
        lines.append("📅 分析时间：" + datetime.now().strftime("%Y年%m月%d日 %H:%M"))

        return "\n".join(lines)


class SimpleRAGSystem:
    """简化的RAG系统"""

    def __init__(self, output_dir: str = None):
        # 输出目录
        if output_dir is None:
            base_dir = config.paths.get("finetune_output", "/home/Liyang/agent/finetune_output")
            self.output_dir = Path(base_dir) / "rag_system"
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化日志
        self._setup_logging()

        # 初始化组件
        self.logger.info("初始化RAG系统组件...")
        self.retriever = KnowledgeBaseRetriever()
        self.feature_extractor = WeatherFeatureExtractor()
        self.response_generator = ResponseGenerator()

        self.logger.info("✅ RAG系统初始化完成")

    def _setup_logging(self):
        """设置日志"""
        log_file = self.output_dir / "rag_system.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger(__name__)

    def process_query(self, query: str, top_k: int = 5) -> Dict:
        """处理查询"""
        self.logger.info(f"处理查询: {query}")

        try:
            # 1. 检索相关文档
            self.logger.info("步骤1: 检索相关文档...")
            retrieved_docs = self.retriever.retrieve(query, top_k=top_k)
            self.logger.info(f"检索到 {len(retrieved_docs)} 个相关文档")

            # 2. 提取天气特征
            self.logger.info("步骤2: 提取天气特征...")
            weather_features = self.feature_extractor.extract(query)
            self.logger.info(f"提取到天气特征: {weather_features}")

            # 3. 生成响应
            self.logger.info("步骤3: 生成响应...")
            response = self.response_generator.generate(query, retrieved_docs, weather_features)

            # 4. 保存结果
            self.logger.info("步骤4: 保存结果...")
            self._save_result(response)

            return response

        except Exception as e:
            self.logger.error(f"处理查询失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

            # 返回错误响应
            return {
                'query': query,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'natural_response': f"抱歉，处理查询时出现错误: {str(e)}"
            }

    def _save_result(self, response: Dict):
        """保存结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = self.output_dir / f"result_{timestamp}.json"

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(response, f, indent=2, ensure_ascii=False)

        self.logger.info(f"结果保存到: {result_file}")

    def interactive_mode(self):
        """交互模式"""
        print("=" * 60)
        print("🤖 气象智能RAG系统 - 交互模式")
        print("=" * 60)
        print("输入 'quit' 退出，输入 'help' 查看帮助")
        print()

        while True:
            try:
                query = input("请输入您的气象查询: ").strip()

                if query.lower() == 'quit':
                    print("感谢使用，再见！")
                    break

                if query.lower() == 'help':
                    self._show_help()
                    continue

                if not query:
                    print("请输入有效的查询")
                    continue

                # 处理查询
                print("\n" + "=" * 40)
                print("🔍 正在处理您的查询...")
                response = self.process_query(query)

                # 显示结果
                print("\n" + "=" * 60)
                print("📋 查询结果:")
                print("=" * 60)
                print(response['natural_response'])
                print("=" * 60)
                print()

            except KeyboardInterrupt:
                print("\n\n程序被中断，退出...")
                break
            except Exception as e:
                print(f"发生错误: {e}")
                continue

    def _show_help(self):
        """显示帮助"""
        help_text = """
        气象智能RAG系统 - 帮助
        
        您可以询问以下类型的问题：
        
        1. 温度相关：
           - "今天温度35℃会有什么影响？"
           - "高温天气需要注意什么？"
        
        2. 降水相关：
           - "明天降雨50mm的预测"
           - "暴雨天气如何防护？"
        
        3. 特殊天气：
           - "台风来了怎么办？"
           - "干旱天气如何应对？"
           - "寒潮天气防护措施"
        
        4. 综合查询：
           - "温度30℃湿度70%的天气情况"
           - "未来三天高温干旱预警"
        
        示例查询：
           - 温度35℃湿度40%的天气情况
           - 明天降雨预测
           - 高温热浪防护措施
           - 台风天气注意事项
        
        命令：
           - help: 显示此帮助信息
           - quit: 退出系统
        
        系统会从知识库中检索相关信息，并结合气象特征给出建议。
        """
        print(help_text)

    def batch_process(self, queries_file: str):
        """批量处理查询"""
        queries_path = Path(queries_file)

        if not queries_path.exists():
            self.logger.error(f"查询文件不存在: {queries_file}")
            return

        # 读取查询
        with open(queries_path, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]

        self.logger.info(f"开始批量处理 {len(queries)} 个查询")

        results = []
        for i, query in enumerate(queries, 1):
            self.logger.info(f"处理查询 {i}/{len(queries)}: {query}")

            try:
                response = self.process_query(query)
                results.append(response)
            except Exception as e:
                self.logger.error(f"处理查询失败: {query}, 错误: {e}")
                results.append({
                    'query': query,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })

        # 保存批量结果
        batch_file = self.output_dir / f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        self.logger.info(f"批量处理完成，结果保存到: {batch_file}")

        # 统计结果
        successful = sum(1 for r in results if 'error' not in r)
        failed = len(queries) - successful

        print(f"\n批量处理统计:")
        print(f"总查询数: {len(queries)}")
        print(f"成功处理: {successful}")
        print(f"处理失败: {failed}")
        print(f"结果文件: {batch_file}")


def main():
    """主函数"""
    # 解析命令行参数
    import argparse

    parser = argparse.ArgumentParser(description="气象智能RAG系统")
    parser.add_argument('--query', type=str, help='直接处理单个查询')
    parser.add_argument('--batch', type=str, help='批量处理查询文件')
    parser.add_argument('--output', type=str, help='输出目录')

    args = parser.parse_args()

    # 创建RAG系统
    rag_system = SimpleRAGSystem(output_dir=args.output)

    if args.query:
        # 处理单个查询
        response = rag_system.process_query(args.query)
        print("\n" + "=" * 60)
        print("📋 查询结果:")
        print("=" * 60)
        print(response['natural_response'])
        print("=" * 60)

    elif args.batch:
        # 批量处理
        rag_system.batch_process(args.batch)

    else:
        # 交互模式
        rag_system.interactive_mode()


if __name__ == "__main__":
    main()