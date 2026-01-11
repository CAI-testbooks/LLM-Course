"""
四智能体气象RAG系统
多智能体协作框架
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
import json
import pickle
from pathlib import Path
import warnings
import logging
import re
from datetime import datetime, timedelta
import asyncio
from collections import defaultdict

warnings.filterwarnings('ignore')

from sentence_transformers import SentenceTransformer, util
from config import config


class BaseAgent:
    """智能体基类"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"Agent.{name}")

    def log(self, message: str, level: str = "info"):
        """日志记录"""
        getattr(self.logger, level)(f"[{self.name}] {message}")

    def validate_input(self, input_data: Any) -> bool:
        """验证输入"""
        return True

    def process(self, **kwargs) -> Dict:
        """处理函数（子类必须实现）"""
        raise NotImplementedError


class RetrievalAgent(BaseAgent):
    """检索智能体 - 负责知识检索"""

    def __init__(self, knowledge_path: str = None):
        super().__init__(
            name="RetrievalAgent",
            description="负责从知识库中检索相关信息，提供科学依据和预警指标"
        )

        self.knowledge_path = knowledge_path or config.paths.get(
            "knowledge_json",
            "/home/Liyang/agent/knowledge_base.json"
        )

        # 加载知识库
        self.knowledge_base = self._load_knowledge_base()

        # 初始化检索模型
        model_name = config.knowledge_config.get(
            'base_model',
            'paraphrase-multilingual-MiniLM-L12-v2'
        )
        self.model = SentenceTransformer(model_name)

        # 准备文档
        self.documents, self.doc_embeddings = self._prepare_documents()

        self.log(f"初始化完成，加载 {len(self.documents)} 个文档")

    def _load_knowledge_base(self) -> Dict:
        """加载知识库"""
        knowledge_path = Path(self.knowledge_path)

        if not knowledge_path.exists():
            self.log(f"警告: 知识库文件不存在: {knowledge_path}", "warning")
            return {"items": []}

        try:
            with open(knowledge_path, 'r', encoding='utf-8') as f:
                knowledge_base = json.load(f)
            return knowledge_base
        except Exception as e:
            self.log(f"加载知识库失败: {e}", "error")
            return {"items": []}

    def _prepare_documents(self) -> Tuple[List[str], np.ndarray]:
        """准备文档和嵌入向量"""
        documents = []

        # 格式化知识库条目
        for item in self.knowledge_base.get('items', []):
            doc_text = self._format_document(item)
            documents.append(doc_text)

        if not documents:
            # 创建示例文档
            documents = [
                "高温热浪: 气温≥35℃时容易导致中暑，建议减少户外活动，多喝水。",
                "暴雨洪水: 短时强降雨可能导致内涝，注意交通安全，避免涉水。",
                "台风防御: 台风天气风力强劲，请固定好门窗，避免外出。",
                "干旱应对: 干旱天气需要节约用水，注意防火，减少户外活动。",
                "寒潮防护: 寒潮天气气温骤降，请注意保暖，预防感冒。"
            ]
            self.log("使用示例文档", "warning")

        # 计算嵌入向量
        embeddings = self.model.encode(
            documents,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return documents, embeddings

    def _format_document(self, item: Dict) -> str:
        """格式化文档"""
        parts = []
        if 'title' in item:
            parts.append(f"标题: {item['title']}")
        if 'category' in item:
            parts.append(f"类别: {item['category']}")
        if 'scientific_basis' in item:
            parts.append(f"科学依据: {item['scientific_basis']}")
        if 'warning_indicators' in item:
            parts.append(f"预警指标: {item['warning_indicators']}")
        return "\n".join(parts)

    def process(self, query: str, top_k: int = 5, **kwargs) -> Dict:
        """检索相关文档"""
        self.log(f"检索查询: {query}")

        if len(self.documents) == 0:
            return {
                'success': False,
                'error': '知识库为空',
                'results': []
            }

        try:
            # 编码查询
            query_embedding = self.model.encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            # 计算相似度
            similarities = util.cos_sim(query_embedding, self.doc_embeddings)[0]

            # 获取top_k结果
            top_indices = torch.topk(
                similarities,
                k=min(top_k, len(self.documents))
            ).indices.tolist()

            # 构建结果
            results = []
            for idx in top_indices:
                similarity = similarities[idx].item()
                doc_text = self.documents[idx]

                # 提取关键信息
                category = "未知"
                title = "无标题"

                # 从文档文本中解析信息
                for line in doc_text.split('\n'):
                    if line.startswith('类别:'):
                        category = line.replace('类别:', '').strip()
                    elif line.startswith('标题:'):
                        title = line.replace('标题:', '').strip()

                results.append({
                    'document': doc_text,
                    'category': category,
                    'title': title,
                    'similarity': similarity,
                    'confidence': min(0.99, similarity * 1.2)  # 置信度增强
                })

            # 按类别分组
            category_groups = defaultdict(list)
            for result in results:
                category_groups[result['category']].append(result)

            # 计算类别权重
            category_scores = {}
            for category, items in category_groups.items():
                category_scores[category] = sum(item['similarity'] for item in items) / len(items)

            self.log(f"检索完成，找到 {len(results)} 个相关文档")

            return {
                'success': True,
                'query': query,
                'results': results,
                'category_scores': dict(category_scores),
                'total_docs': len(self.documents),
                'top_categories': sorted(category_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            }

        except Exception as e:
            self.log(f"检索失败: {e}", "error")
            return {
                'success': False,
                'error': str(e),
                'results': []
            }


class AnalysisAgent(BaseAgent):
    """分析智能体 - 负责气象特征提取和风险评估"""

    def __init__(self):
        super().__init__(
            name="AnalysisAgent",
            description="负责分析气象特征，进行风险评估和趋势预测"
        )

        # 初始化特征提取规则
        self.feature_rules = self._init_feature_rules()

        # 风险评估模型
        self.risk_levels = {
            'low': {'min': 0, 'max': 3, 'color': '🟢', 'description': '低风险'},
            'medium': {'min': 4, 'max': 6, 'color': '🟡', 'description': '中风险'},
            'high': {'min': 7, 'max': 9, 'color': '🟠', 'description': '高风险'},
            'extreme': {'min': 10, 'max': 12, 'color': '🔴', 'description': '极高风险'}
        }

        self.log("初始化完成")

    def _init_feature_rules(self) -> Dict:
        """初始化特征提取规则"""
        return {
            'temperature': {
                'patterns': [
                    r'温度\s*([0-9]+\.?[0-9]*)\s*℃',
                    r'([0-9]+\.?[0-9]*)\s*℃',
                    r'气温\s*([0-9]+\.?[0-9]*)度'
                ],
                'unit': '℃',
                'risk_weight': 1.5
            },
            'humidity': {
                'patterns': [
                    r'湿度\s*([0-9]+\.?[0-9]*)\s*%',
                    r'([0-9]+\.?[0-9]*)\s*%湿度'
                ],
                'unit': '%',
                'risk_weight': 1.0
            },
            'precipitation': {
                'patterns': [
                    r'降雨\s*([0-9]+\.?[0-9]*)\s*mm',
                    r'降水\s*([0-9]+\.?[0-9]*)\s*毫米',
                    r'雨量\s*([0-9]+\.?[0-9]*)'
                ],
                'unit': 'mm',
                'risk_weight': 1.8
            },
            'wind': {
                'patterns': [
                    r'风速\s*([0-9]+\.?[0-9]*)\s*m/s',
                    r'风力\s*([0-9]+\.?[0-9]*)\s*级'
                ],
                'unit': 'm/s',
                'risk_weight': 1.3
            }
        }

    def extract_features(self, query: str) -> Dict:
        """提取气象特征"""
        features = {'raw_features': {}, 'keywords': []}

        # 提取数值特征
        for feature_name, rule in self.feature_rules.items():
            for pattern in rule['patterns']:
                matches = re.findall(pattern, query)
                if matches:
                    values = [float(match) for match in matches if self._is_number(match)]
                    if values:
                        avg_value = sum(values) / len(values)
                        features['raw_features'][feature_name] = {
                            'value': avg_value,
                            'unit': rule['unit'],
                            'risk_weight': rule['risk_weight']
                        }
                        break

        # 提取关键词
        weather_keywords = {
            '高温': 'heat',
            '热浪': 'heatwave',
            '暴雨': 'heavy_rain',
            '台风': 'typhoon',
            '干旱': 'drought',
            '寒潮': 'cold_wave',
            '大风': 'strong_wind',
            '冰雹': 'hail',
            '雷电': 'lightning',
            '雾霾': 'haze',
            '沙尘': 'sandstorm'
        }

        for keyword, key in weather_keywords.items():
            if keyword in query:
                features['keywords'].append({
                    'keyword': keyword,
                    'key': key,
                    'risk_level': self._get_keyword_risk(keyword)
                })

        # 提取时间信息
        time_keywords = {
            '今天': 'today',
            '明天': 'tomorrow',
            '后天': 'day_after_tomorrow',
            '本周': 'this_week',
            '周末': 'weekend',
            '未来三天': 'next_3_days',
            '下周': 'next_week',
            '近期': 'recent'
        }

        for keyword, key in time_keywords.items():
            if keyword in query:
                features['time_period'] = key
                break

        self.log(f"提取到特征: {features}")
        return features

    def _is_number(self, s: str) -> bool:
        """判断是否为数字"""
        try:
            float(s)
            return True
        except ValueError:
            return False

    def _get_keyword_risk(self, keyword: str) -> str:
        """获取关键词风险等级"""
        risk_map = {
            '高温': 'high', '热浪': 'extreme', '暴雨': 'high',
            '台风': 'extreme', '干旱': 'medium', '寒潮': 'medium',
            '大风': 'medium', '冰雹': 'high', '雷电': 'high',
            '雾霾': 'low', '沙尘': 'medium'
        }
        return risk_map.get(keyword, 'low')

    def assess_risk(self, features: Dict, retrieval_results: Dict = None) -> Dict:
        """风险评估"""
        risk_score = 0
        risk_factors = []

        # 1. 数值特征风险评估
        for feature_name, feature_data in features.get('raw_features', {}).items():
            value = feature_data['value']
            weight = feature_data['risk_weight']

            # 根据特征值计算风险
            if feature_name == 'temperature':
                if value >= 35:
                    risk_score += 3 * weight
                    risk_factors.append(f"高温({value}℃)")
                elif value >= 30:
                    risk_score += 2 * weight
                    risk_factors.append(f"炎热({value}℃)")

            elif feature_name == 'precipitation':
                if value >= 50:
                    risk_score += 3 * weight
                    risk_factors.append(f"暴雨({value}mm)")
                elif value >= 25:
                    risk_score += 2 * weight
                    risk_factors.append(f"大雨({value}mm)")

            elif feature_name == 'wind':
                if value >= 10.8:  # 6级风以上
                    risk_score += 2 * weight
                    risk_factors.append(f"大风({value}m/s)")

        # 2. 关键词风险评估
        for keyword_data in features.get('keywords', []):
            risk_level = keyword_data['risk_level']
            keyword = keyword_data['keyword']

            if risk_level == 'extreme':
                risk_score += 4
                risk_factors.append(f"{keyword}(极高风险)")
            elif risk_level == 'high':
                risk_score += 3
                risk_factors.append(f"{keyword}(高风险)")
            elif risk_level == 'medium':
                risk_score += 2
                risk_factors.append(f"{keyword}(中风险)")
            else:
                risk_score += 1
                risk_factors.append(f"{keyword}(低风险)")

        # 3. 结合检索结果的类别风险
        if retrieval_results and retrieval_results.get('success'):
            top_categories = retrieval_results.get('top_categories', [])
            for category, score in top_categories:
                if any(high_risk in category for high_risk in ['高温', '台风', '暴雨', '干旱']):
                    risk_score += score * 2
                    risk_factors.append(f"相关类别: {category}")

        # 确定风险等级
        risk_level = 'low'
        level_info = None

        for level_name, level_range in self.risk_levels.items():
            if level_range['min'] <= risk_score <= level_range['max']:
                risk_level = level_name
                level_info = level_range
                break
        else:
            # 如果超过最大值，设为最高风险
            risk_level = 'extreme'
            level_info = self.risk_levels['extreme']

        risk_assessment = {
            'risk_score': round(risk_score, 2),
            'risk_level': risk_level,
            'level_info': level_info,
            'risk_factors': risk_factors,
            'risk_components': {
                'feature_risk': round(risk_score * 0.6, 2),
                'keyword_risk': round(risk_score * 0.3, 2),
                'category_risk': round(risk_score * 0.1, 2)
            }
        }

        self.log(f"风险评估完成: {risk_assessment}")
        return risk_assessment

    def generate_analysis_report(self, query: str, features: Dict,
                                 risk_assessment: Dict) -> Dict:
        """生成分析报告"""
        report = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'feature_analysis': {},
            'risk_assessment': risk_assessment,
            'trend_analysis': {},
            'confidence': 0.85
        }

        # 特征分析
        if features.get('raw_features'):
            report['feature_analysis']['numerical_features'] = features['raw_features']

        if features.get('keywords'):
            report['feature_analysis']['detected_keywords'] = features['keywords']

        if features.get('time_period'):
            report['feature_analysis']['time_period'] = features['time_period']

        # 趋势分析（模拟）
        trend_indicators = []

        if 'temperature' in features.get('raw_features', {}):
            temp = features['raw_features']['temperature']['value']
            if temp > 30:
                trend_indicators.append("温度呈上升趋势，可能发展为热浪天气")

        if any(k['key'] == 'heavy_rain' for k in features.get('keywords', [])):
            trend_indicators.append("降水条件具备，可能发展为持续性降雨")

        report['trend_analysis']['indicators'] = trend_indicators
        report['trend_analysis']['prediction_horizon'] = "未来24-48小时"

        return report

    def process(self, query: str, retrieval_results: Dict = None, **kwargs) -> Dict:
        """处理分析任务"""
        self.log(f"分析查询: {query}")

        try:
            # 1. 提取特征
            features = self.extract_features(query)

            # 2. 风险评估
            risk_assessment = self.assess_risk(features, retrieval_results)

            # 3. 生成分析报告
            analysis_report = self.generate_analysis_report(
                query, features, risk_assessment
            )

            analysis_report['success'] = True
            return analysis_report

        except Exception as e:
            self.log(f"分析失败: {e}", "error")
            return {
                'success': False,
                'error': str(e),
                'query': query
            }


class DecisionAgent(BaseAgent):
    """决策智能体 - 负责生成应对建议和措施"""

    def __init__(self):
        super().__init__(
            name="DecisionAgent",
            description="根据分析和检索结果，生成具体应对建议和决策方案"
        )

        # 决策规则库
        self.decision_rules = self._init_decision_rules()

        # 建议模板
        self.recommendation_templates = {
            'high_temperature': [
                "避免在高温时段(10:00-16:00)进行户外活动",
                "穿着宽松、透气的浅色衣物",
                "及时补充水分，每天至少饮水2-3升",
                "使用防晒霜(SPF30+)，佩戴太阳镜和遮阳帽",
                "关注老人、儿童和慢性病患者的健康状况",
                "如出现头晕、恶心等中暑症状，立即到阴凉处休息并就医"
            ],
            'heavy_rain': [
                "关注气象预警，避免前往低洼地带",
                "驾车时注意减速慢行，保持安全车距",
                "避免在树下、广告牌下停留，防止雷击",
                "检查房屋排水系统，防止雨水倒灌",
                "准备应急照明和通讯设备",
                "如遇积水路段，不要强行通过"
            ],
            'typhoon': [
                "加固门窗，移除阳台上的易坠落物品",
                "储备3天以上的食物、水和药品",
                "避免外出，如需外出请远离海岸和山区",
                "关注官方发布的台风路径和预警信息",
                "准备应急电源，保持通讯畅通",
                "台风过后注意检查房屋安全，防范次生灾害"
            ],
            'drought': [
                "节约用水，优先保证生活用水",
                "调整农业灌溉时间，避免中午高温时段",
                "注意防火，不要在林区和野外用火",
                "做好个人防护，防止皮肤干燥开裂",
                "关注水库蓄水情况和供水通知",
                "考虑雨水收集和中水回用"
            ],
            'cold_wave': [
                "注意保暖，特别是头部、手部和脚部",
                "使用取暖设备时注意通风，防止一氧化碳中毒",
                "老人、儿童和体弱者减少外出",
                "注意水管防冻，防止爆裂",
                "适当增加高热量食物摄入",
                "关注天气预报，及时添加衣物"
            ],
            'general': [
                "关注当地气象部门的最新预报和预警",
                "根据天气变化及时调整出行计划",
                "保持通讯畅通，随时了解天气信息",
                "准备必要的应急物资",
                "学习基本的防灾减灾知识"
            ]
        }

        self.log("初始化完成")

    def _init_decision_rules(self) -> Dict:
        """初始化决策规则"""
        return {
            'heatwave': {
                'conditions': ['temperature>=35', 'has_heatwave'],
                'priority': 1,
                'action_type': 'immediate'
            },
            'heavy_rain_alert': {
                'conditions': ['precipitation>=50', 'has_heavy_rain'],
                'priority': 1,
                'action_type': 'immediate'
            },
            'typhoon_warning': {
                'conditions': ['has_typhoon', 'wind>=10.8'],
                'priority': 1,
                'action_type': 'emergency'
            },
            'drought_alert': {
                'conditions': ['has_drought', 'humidity<=30'],
                'priority': 2,
                'action_type': 'monitor'
            },
            'cold_protection': {
                'conditions': ['temperature<=10', 'has_cold_wave'],
                'priority': 2,
                'action_type': 'preventive'
            }
        }

    def generate_decisions(self, analysis_report: Dict,
                           retrieval_results: Dict) -> List[Dict]:
        """生成决策建议"""
        decisions = []

        # 获取分析结果
        features = analysis_report.get('feature_analysis', {})
        risk_assessment = analysis_report.get('risk_assessment', {})
        risk_level = risk_assessment.get('risk_level', 'low')

        # 确定需要应对的天气类型
        weather_types = set()

        # 从关键词中提取
        for keyword_data in features.get('detected_keywords', []):
            key = keyword_data['key']
            if key in ['heat', 'heatwave']:
                weather_types.add('high_temperature')
            elif key in ['heavy_rain', 'typhoon']:
                weather_types.add(key)
            elif key in ['drought', 'cold_wave']:
                weather_types.add(key)

        # 从数值特征中判断
        numerical_features = features.get('numerical_features', {})
        if 'temperature' in numerical_features:
            temp = numerical_features['temperature']['value']
            if temp >= 35:
                weather_types.add('high_temperature')
            elif temp <= 10:
                weather_types.add('cold_wave')

        if 'precipitation' in numerical_features:
            precip = numerical_features['precipitation']['value']
            if precip >= 50:
                weather_types.add('heavy_rain')

        # 如果没有特定天气类型，使用通用建议
        if not weather_types:
            weather_types.add('general')

        # 根据风险等级调整建议强度
        priority_map = {
            'extreme': '紧急应对',
            'high': '高度重视',
            'medium': '加强防范',
            'low': '正常关注'
        }

        priority = priority_map.get(risk_level, '正常关注')

        # 生成具体决策
        for weather_type in weather_types:
            if weather_type in self.recommendation_templates:
                recommendations = self.recommendation_templates[weather_type]

                # 根据风险等级调整建议数量
                if risk_level in ['low', 'medium']:
                    recommendations = recommendations[:3]
                elif risk_level == 'high':
                    recommendations = recommendations[:5]
                # extreme风险等级使用所有建议

                decisions.append({
                    'weather_type': weather_type,
                    'priority': priority,
                    'recommendations': recommendations,
                    'applicable_conditions': self._get_applicable_conditions(weather_type)
                })

        # 从检索结果中提取额外建议
        if retrieval_results.get('success'):
            top_results = retrieval_results.get('results', [])[:2]
            for result in top_results:
                doc_text = result.get('document', '')
                # 从文档中提取关键建议
                if '应对:' in doc_text:
                    response_part = doc_text.split('应对:')[1]
                    key_points = [p.strip() for p in response_part.split('。') if p.strip()]

                    if key_points:
                        decisions.append({
                            'weather_type': result.get('category', '通用'),
                            'priority': '知识库建议',
                            'recommendations': key_points[:3],
                            'source': '知识库',
                            'confidence': result.get('confidence', 0.7)
                        })

        self.log(f"生成 {len(decisions)} 个决策建议")
        return decisions

    def _get_applicable_conditions(self, weather_type: str) -> List[str]:
        """获取适用条件"""
        conditions_map = {
            'high_temperature': ['气温≥35℃', '相对湿度≥60%', '连续高温≥3天'],
            'heavy_rain': ['小时降雨量≥50mm', '持续降雨≥3小时', '伴有雷电'],
            'typhoon': ['风力≥10级', '伴有暴雨', '风暴潮预警'],
            'drought': ['连续无降水≥15天', '土壤湿度≤30%', '水库蓄水不足'],
            'cold_wave': ['24小时降温≥8℃', '最低气温≤0℃', '伴有大风']
        }
        return conditions_map.get(weather_type, ['通用天气条件'])

    def generate_action_plan(self, decisions: List[Dict]) -> Dict:
        """生成行动方案"""
        action_plan = {
            'immediate_actions': [],
            'short_term_actions': [],
            'monitoring_actions': [],
            'preparedness_actions': []
        }

        for decision in decisions:
            weather_type = decision['weather_type']
            priority = decision['priority']
            recommendations = decision['recommendations']

            if priority in ['紧急应对', '高度重视']:
                action_plan['immediate_actions'].extend(recommendations[:2])
                action_plan['short_term_actions'].extend(recommendations[2:4])
            elif priority == '加强防范':
                action_plan['short_term_actions'].extend(recommendations[:3])
                action_plan['monitoring_actions'].extend(recommendations[3:])
            else:
                action_plan['preparedness_actions'].extend(recommendations[:3])

        # 去重
        for key in action_plan:
            action_plan[key] = list(set(action_plan[key]))

        return action_plan

    def process(self, analysis_report: Dict, retrieval_results: Dict, **kwargs) -> Dict:
        """处理决策任务"""
        self.log(f"生成决策建议")

        try:
            # 1. 生成决策建议
            decisions = self.generate_decisions(analysis_report, retrieval_results)

            # 2. 生成行动方案
            action_plan = self.generate_action_plan(decisions)

            # 3. 生成决策报告
            decision_report = {
                'success': True,
                'decisions': decisions,
                'action_plan': action_plan,
                'summary': self._generate_decision_summary(decisions),
                'timestamp': datetime.now().isoformat()
            }

            return decision_report

        except Exception as e:
            self.log(f"决策生成失败: {e}", "error")
            return {
                'success': False,
                'error': str(e)
            }

    def _generate_decision_summary(self, decisions: List[Dict]) -> str:
        """生成决策摘要"""
        if not decisions:
            return "当前天气条件正常，建议关注常规天气预报"

        summary_parts = []
        for decision in decisions:
            weather_type = decision['weather_type']
            priority = decision['priority']

            if weather_type == 'high_temperature':
                summary_parts.append(f"高温天气，{priority}")
            elif weather_type == 'heavy_rain':
                summary_parts.append(f"暴雨天气，{priority}")
            elif weather_type == 'typhoon':
                summary_parts.append(f"台风天气，{priority}")
            elif weather_type == 'drought':
                summary_parts.append(f"干旱天气，{priority}")
            elif weather_type == 'cold_wave':
                summary_parts.append(f"寒潮天气，{priority}")

        return "；".join(summary_parts) if summary_parts else "天气条件复杂，请关注详细建议"


class CoordinatorAgent(BaseAgent):
    """协调智能体 - 负责协调其他智能体工作"""

    def __init__(self, agents: Dict[str, BaseAgent]):
        super().__init__(
            name="CoordinatorAgent",
            description="协调和管理各智能体的工作流程，整合最终结果"
        )

        self.agents = agents
        self.workflow_status = {}
        self.results_cache = {}

        self.log(f"初始化完成，管理 {len(agents)} 个智能体")

    async def execute_workflow(self, query: str) -> Dict:
        """执行工作流程"""
        self.log(f"开始执行工作流程，查询: {query}")

        workflow_id = f"wf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.workflow_status[workflow_id] = {
            'query': query,
            'start_time': datetime.now().isoformat(),
            'status': 'running',
            'agents_status': {}
        }

        try:
            # 1. 检索阶段
            self.log("阶段1: 知识检索")
            retrieval_agent = self.agents.get('retrieval')
            if retrieval_agent:
                retrieval_result = await self._execute_agent_task(
                    retrieval_agent, 'retrieval', workflow_id, query=query
                )
                self.results_cache['retrieval'] = retrieval_result
            else:
                retrieval_result = {'success': False, 'error': '检索智能体未配置'}

            # 2. 分析阶段
            self.log("阶段2: 特征分析与风险评估")
            analysis_agent = self.agents.get('analysis')
            if analysis_agent:
                analysis_result = await self._execute_agent_task(
                    analysis_agent, 'analysis', workflow_id,
                    query=query, retrieval_results=retrieval_result
                )
                self.results_cache['analysis'] = analysis_result
            else:
                analysis_result = {'success': False, 'error': '分析智能体未配置'}

            # 3. 决策阶段
            self.log("阶段3: 决策建议生成")
            decision_agent = self.agents.get('decision')
            if decision_agent:
                decision_result = await self._execute_agent_task(
                    decision_agent, 'decision', workflow_id,
                    analysis_report=analysis_result,
                    retrieval_results=retrieval_result
                )
                self.results_cache['decision'] = decision_result
            else:
                decision_result = {'success': False, 'error': '决策智能体未配置'}

            # 4. 整合结果
            self.log("阶段4: 结果整合")
            final_result = self._integrate_results(
                query, retrieval_result, analysis_result, decision_result
            )

            # 更新工作流状态
            self.workflow_status[workflow_id].update({
                'end_time': datetime.now().isoformat(),
                'status': 'completed',
                'final_result': final_result.get('success', False)
            })

            self.log(f"工作流程完成: {workflow_id}")
            return final_result

        except Exception as e:
            self.log(f"工作流程执行失败: {e}", "error")

            self.workflow_status[workflow_id].update({
                'end_time': datetime.now().isoformat(),
                'status': 'failed',
                'error': str(e)
            })

            return {
                'success': False,
                'error': f"工作流程执行失败: {str(e)}",
                'query': query,
                'workflow_id': workflow_id
            }

    async def _execute_agent_task(self, agent: BaseAgent, agent_name: str,
                                  workflow_id: str, **kwargs) -> Dict:
        """执行智能体任务"""
        try:
            start_time = datetime.now()

            # 记录开始状态
            self.workflow_status[workflow_id]['agents_status'][agent_name] = {
                'status': 'running',
                'start_time': start_time.isoformat()
            }

            # 执行任务
            result = agent.process(**kwargs)

            # 记录结束状态
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            self.workflow_status[workflow_id]['agents_status'][agent_name].update({
                'status': 'completed' if result.get('success') else 'failed',
                'end_time': end_time.isoformat(),
                'duration': duration,
                'success': result.get('success', False)
            })

            agent.log(f"任务完成，耗时: {duration:.2f}秒")
            return result

        except Exception as e:
            agent.log(f"任务执行失败: {e}", "error")

            self.workflow_status[workflow_id]['agents_status'][agent_name] = {
                'status': 'failed',
                'error': str(e),
                'end_time': datetime.now().isoformat()
            }

            return {
                'success': False,
                'error': f"{agent_name}执行失败: {str(e)}"
            }

    def _integrate_results(self, query: str, retrieval_result: Dict,
                           analysis_result: Dict, decision_result: Dict) -> Dict:
        """整合所有结果"""
        final_result = {
            'success': all([
                retrieval_result.get('success', False),
                analysis_result.get('success', False),
                decision_result.get('success', False)
            ]),
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'components': {
                'retrieval': retrieval_result.get('success', False),
                'analysis': analysis_result.get('success', False),
                'decision': decision_result.get('success', False)
            }
        }

        if final_result['success']:
            # 整合成功结果
            final_result.update({
                'knowledge_retrieval': {
                    'total_docs': retrieval_result.get('total_docs', 0),
                    'relevant_docs': len(retrieval_result.get('results', [])),
                    'top_categories': retrieval_result.get('top_categories', [])
                },
                'risk_assessment': analysis_result.get('risk_assessment', {}),
                'decisions': decision_result.get('decisions', []),
                'action_plan': decision_result.get('action_plan', {}),
                'summary': decision_result.get('summary', '')
            })

            # 生成最终响应
            final_result['response'] = self._generate_final_response(
                query, analysis_result, decision_result
            )

            # 计算综合置信度
            confidence_sources = []
            if retrieval_result.get('results'):
                conf = retrieval_result['results'][0].get('confidence', 0) if retrieval_result['results'] else 0
                confidence_sources.append(conf)

            if 'confidence' in analysis_result:
                confidence_sources.append(analysis_result['confidence'])

            final_result['confidence'] = sum(confidence_sources) / len(
                confidence_sources) if confidence_sources else 0.7

        else:
            # 处理失败情况
            errors = []
            if not retrieval_result.get('success'):
                errors.append(f"检索失败: {retrieval_result.get('error')}")
            if not analysis_result.get('success'):
                errors.append(f"分析失败: {analysis_result.get('error')}")
            if not decision_result.get('success'):
                errors.append(f"决策失败: {decision_result.get('error')}")

            final_result['errors'] = errors
            final_result['response'] = f"抱歉，处理您的查询时出现错误: {'; '.join(errors)}"

        return final_result

    def _generate_final_response(self, query: str, analysis_result: Dict,
                                 decision_result: Dict) -> str:
        """生成最终响应文本"""
        risk_assessment = analysis_result.get('risk_assessment', {})
        decisions = decision_result.get('decisions', [])
        action_plan = decision_result.get('action_plan', {})

        lines = []

        # 标题
        lines.append("=" * 60)
        lines.append(f"🌤️ 气象智能分析报告 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("=" * 60)
        lines.append(f"📝 查询: {query}")
        lines.append("")

        # 风险评估
        risk_level = risk_assessment.get('risk_level', 'low')
        level_info = risk_assessment.get('level_info', {})

        lines.append("⚠️ **风险评估**")
        lines.append(f"- 风险等级: {level_info.get('description', '未知')} {level_info.get('color', '')}")
        lines.append(f"- 风险评分: {risk_assessment.get('risk_score', 0)}/12")

        risk_factors = risk_assessment.get('risk_factors', [])
        if risk_factors:
            lines.append(f"- 主要风险因素:")
            for factor in risk_factors[:3]:
                lines.append(f"  • {factor}")

        lines.append("")

        # 决策建议
        lines.append("💡 **决策建议**")
        for i, decision in enumerate(decisions[:3], 1):
            weather_type = decision['weather_type']
            priority = decision['priority']
            lines.append(f"{i}. {weather_type} - {priority}")

            for rec in decision.get('recommendations', [])[:2]:
                lines.append(f"   ✓ {rec}")

        lines.append("")

        # 行动方案
        lines.append("🚀 **行动方案**")

        if action_plan.get('immediate_actions'):
            lines.append("立即行动:")
            for action in action_plan['immediate_actions'][:3]:
                lines.append(f"• {action}")

        if action_plan.get('short_term_actions'):
            lines.append("短期行动:")
            for action in action_plan['short_term_actions'][:3]:
                lines.append(f"• {action}")

        lines.append("")
        lines.append("=" * 60)
        lines.append("🔬 分析系统：四智能体协作框架")
        lines.append("  1. 检索智能体 - 知识库检索")
        lines.append("  2. 分析智能体 - 特征分析与风险评估")
        lines.append("  3. 决策智能体 - 建议与方案生成")
        lines.append("  4. 协调智能体 - 工作流程管理")
        lines.append("=" * 60)

        return "\n".join(lines)

    def get_workflow_status(self, workflow_id: str = None) -> Dict:
        """获取工作流状态"""
        if workflow_id:
            return self.workflow_status.get(workflow_id, {})
        return self.workflow_status

    def clear_cache(self):
        """清空缓存"""
        self.results_cache.clear()
        self.log("缓存已清空")


class MultiAgentSystem:
    """多智能体系统"""

    def __init__(self, output_dir: str = None):
        # 设置输出目录
        if output_dir is None:
            base_dir = config.paths.get("finetune_output", "/home/Liyang/agent/finetune_output")
            self.output_dir = Path(base_dir) / "multi_agent_system"
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 设置日志
        self._setup_logging()

        # 初始化智能体
        self.logger.info("初始化多智能体系统...")
        self.agents = self._initialize_agents()
        self.coordinator = CoordinatorAgent(self.agents)

        self.logger.info("✅ 多智能体系统初始化完成")

    def _setup_logging(self):
        """设置日志"""
        log_file = self.output_dir / "multi_agent_system.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger("MultiAgentSystem")

    def _initialize_agents(self) -> Dict[str, BaseAgent]:
        """初始化所有智能体"""
        agents = {}

        # 1. 检索智能体
        try:
            retrieval_agent = RetrievalAgent()
            agents['retrieval'] = retrieval_agent
            self.logger.info(f"✅ 初始化检索智能体: {retrieval_agent.name}")
        except Exception as e:
            self.logger.error(f"初始化检索智能体失败: {e}")

        # 2. 分析智能体
        try:
            analysis_agent = AnalysisAgent()
            agents['analysis'] = analysis_agent
            self.logger.info(f"✅ 初始化分析智能体: {analysis_agent.name}")
        except Exception as e:
            self.logger.error(f"初始化分析智能体失败: {e}")

        # 3. 决策智能体
        try:
            decision_agent = DecisionAgent()
            agents['decision'] = decision_agent
            self.logger.info(f"✅ 初始化决策智能体: {decision_agent.name}")
        except Exception as e:
            self.logger.error(f"初始化决策智能体失败: {e}")

        return agents

    async def process_query_async(self, query: str) -> Dict:
        """异步处理查询"""
        return await self.coordinator.execute_workflow(query)

    def process_query(self, query: str) -> Dict:
        """同步处理查询"""
        try:
            # 创建事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # 执行异步任务
            result = loop.run_until_complete(self.process_query_async(query))
            loop.close()

            return result
        except Exception as e:
            self.logger.error(f"处理查询失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'query': query
            }

    def save_result(self, result: Dict):
        """保存结果"""
        if not result.get('success'):
            self.logger.warning(f"结果不成功，不保存: {result.get('error')}")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = self.output_dir / f"result_{timestamp}.json"

        # 保存完整结果
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # 保存响应文本
        response_file = self.output_dir / f"response_{timestamp}.txt"
        with open(response_file, 'w', encoding='utf-8') as f:
            f.write(result.get('response', ''))

        self.logger.info(f"结果保存到: {result_file}")
        self.logger.info(f"响应保存到: {response_file}")

    def batch_process(self, queries: List[str]) -> List[Dict]:
        """批量处理查询"""
        self.logger.info(f"批量处理 {len(queries)} 个查询")

        results = []

        for i, query in enumerate(queries, 1):
            self.logger.info(f"处理查询 {i}/{len(queries)}: {query}")

            try:
                result = self.process_query(query)
                results.append(result)

                # 保存每个结果
                if result.get('success'):
                    self.save_result(result)

                # 进度报告
                if i % 5 == 0:
                    success_count = sum(1 for r in results if r.get('success'))
                    self.logger.info(f"进度: {i}/{len(queries)}, 成功: {success_count}")

            except Exception as e:
                self.logger.error(f"处理查询失败: {query}, 错误: {e}")
                results.append({
                    'query': query,
                    'success': False,
                    'error': str(e)
                })

        # 保存批量结果摘要
        batch_summary = {
            'total_queries': len(queries),
            'successful': sum(1 for r in results if r.get('success')),
            'failed': sum(1 for r in results if not r.get('success')),
            'avg_confidence': np.mean([
                r.get('confidence', 0) for r in results if r.get('success')
            ]) if any(r.get('success') for r in results) else 0,
            'timestamp': datetime.now().isoformat()
        }

        batch_file = self.output_dir / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(batch_summary, f, indent=2, ensure_ascii=False)

        self.logger.info(f"批量处理完成，摘要保存到: {batch_file}")
        self.logger.info(f"成功: {batch_summary['successful']}, 失败: {batch_summary['failed']}")

        return results

    def interactive_mode(self):
        """交互模式"""
        print("=" * 60)
        print("🤖 四智能体气象分析系统")
        print("=" * 60)
        print("系统架构:")
        print("  1. 检索智能体 - 从知识库检索相关信息")
        print("  2. 分析智能体 - 提取特征并进行风险评估")
        print("  3. 决策智能体 - 生成应对建议和行动方案")
        print("  4. 协调智能体 - 管理整个工作流程")
        print("=" * 60)
        print("输入 'quit' 退出，输入 'help' 查看帮助，输入 'status' 查看状态")
        print()

        while True:
            try:
                user_input = input("请输入气象查询: ").strip()

                if user_input.lower() == 'quit':
                    print("感谢使用，再见！")
                    break

                if user_input.lower() == 'help':
                    self._show_help()
                    continue

                if user_input.lower() == 'status':
                    self._show_system_status()
                    continue

                if not user_input:
                    print("请输入有效的查询")
                    continue

                # 处理查询
                print("\n" + "=" * 40)
                print("🚀 启动四智能体协作流程...")

                result = self.process_query(user_input)

                if result.get('success'):
                    # 显示响应
                    print("\n" + "=" * 60)
                    print(result.get('response', '无响应'))
                    print("=" * 60)

                    # 保存结果
                    self.save_result(result)
                else:
                    print(f"\n❌ 处理失败: {result.get('error', '未知错误')}")

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
        四智能体气象分析系统 - 帮助

        系统架构:
          检索智能体: 从知识库中检索相关气象知识
          分析智能体: 分析气象特征，进行风险评估
          决策智能体: 生成应对建议和行动方案
          协调智能体: 管理整个工作流程

        查询示例:
          1. 温度查询:
            - "今天温度35℃会有什么影响？"
            - "高温40度需要注意什么？"

          2. 降水查询:
            - "明天降雨50mm如何应对？"
            - "暴雨天气安全指南"

          3. 特殊天气:
            - "台风来了怎么办？"
            - "干旱天气应对措施"
            - "寒潮来袭如何防护？"

          4. 综合查询:
            - "温度30℃湿度80%风速10m/s"
            - "未来三天高温暴雨预警"

        命令:
          - help: 显示此帮助信息
          - status: 查看系统状态
          - quit: 退出系统
        """
        print(help_text)

    def _show_system_status(self):
        """显示系统状态"""
        status = {
            'agents_count': len(self.agents),
            'agents': list(self.agents.keys()),
            'coordinator': self.coordinator.name if hasattr(self, 'coordinator') else '未初始化',
            'workflows_count': len(self.coordinator.get_workflow_status()) if hasattr(self, 'coordinator') else 0,
            'output_dir': str(self.output_dir)
        }

        print("\n系统状态:")
        for key, value in status.items():
            print(f"  {key}: {value}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="四智能体气象分析系统")
    parser.add_argument('--query', type=str, help='直接处理单个查询')
    parser.add_argument('--batch', type=str, help='批量处理查询文件')
    parser.add_argument('--output', type=str, help='输出目录')
    parser.add_argument('--interactive', action='store_true', help='交互模式')

    args = parser.parse_args()

    # 创建多智能体系统
    system = MultiAgentSystem(output_dir=args.output)

    if args.query:
        # 处理单个查询
        print(f"处理查询: {args.query}")
        result = system.process_query(args.query)

        if result.get('success'):
            print("\n" + "=" * 60)
            print(result.get('response', '无响应'))
            print("=" * 60)
        else:
            print(f"处理失败: {result.get('error')}")

    elif args.batch:
        # 批量处理
        with open(args.batch, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]

        system.batch_process(queries)

    elif args.interactive or (not args.query and not args.batch):
        # 交互模式（默认）
        system.interactive_mode()


if __name__ == "__main__":
    main()