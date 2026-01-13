"""
基于DSPy的LLM Agent实现
"""
import dspy
from dspy.teleprompt import BootstrapFewShot
from loguru import logger
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

class DataAnalysisSignature(dspy.Signature):
    """数据分析任务签名"""
    task_description = dspy.InputField(desc="分析任务描述")
    data_context = dspy.InputField(desc="数据上下文信息")
    analysis_type = dspy.InputField(desc="分析类型")
    
    analysis_steps = dspy.OutputField(desc="分析步骤")
    python_code = dspy.OutputField(desc="执行的Python代码")
    insights = dspy.OutputField(desc="数据洞察")

class FeatureEngineeringSignature(dspy.Signature):
    """特征工程签名"""
    data_description = dspy.InputField(desc="数据描述")
    columns = dspy.InputField(desc="数据列名")
    target_column = dspy.InputField(desc="目标列", optional=True)
    problem_type = dspy.InputField(desc="问题类型")
    
    feature_ideas = dspy.OutputField(desc="特征工程想法")
    transformation_code = dspy.OutputField(desc="转换代码")
    recommendations = dspy.OutputField(desc="特征选择建议")

class VisualizationSignature(dspy.Signature):
    """可视化签名"""
    data_info = dspy.InputField(desc="数据信息")
    analysis_goal = dspy.InputField(desc="分析目标")
    plot_types = dspy.InputField(desc="可选的图表类型")
    
    visualization_plan = dspy.OutputField(desc="可视化方案")
    plotly_code = dspy.OutputField(desc="Plotly可视化代码")
    insights_from_viz = dspy.OutputField(desc="可视化洞察")

class StatisticalTestSignature(dspy.Signature):
    """统计检验签名"""
    data_summary = dspy.InputField(desc="数据摘要")
    test_objective = dspy.InputField(desc="检验目标")
    hypotheses = dspy.InputField(desc="假设")
    
    appropriate_test = dspy.OutputField(desc="合适的统计检验")
    test_code = dspy.OutputField(desc="检验代码")
    interpretation = dspy.OutputField(desc="结果解释")

class LLMAgent:
    """LLM Agent核心类"""
    
    def __init__(self, config):
        """初始化LLM Agent"""
        self.config = config
        # 设置API密钥到环境变量
        import os
        if config.llm_provider == "openai":
            if config.llm_api_key:
                os.environ["OPENAI_API_KEY"] = config.llm_api_key
            # 新版本DSPy配置方式
            dspy.configure(
                lm="openai/gpt-3.5-turbo",  # 或者使用 config.llm_model
                max_tokens=config.max_tokens,
                temperature=config.temperature
            )
        elif config.llm_provider == "anthropic":
            if config.llm_api_key:
                os.environ["ANTHROPIC_API_KEY"] = config.llm_api_key
            # Anthropic配置
            dspy.configure(
                lm="claude-3-sonnet-20240229",
                max_tokens=config.max_tokens
            )
        else:
            # 本地模型配置
            dspy.configure(
                lm="local:gpt2",  # 修改为适合本地模型的格式
                api_base=config.llm_api_base
            )
        
        # 初始化DSPy模块
        self.data_analyzer = dspy.ChainOfThought(DataAnalysisSignature)
        self.feature_engineer = dspy.ChainOfThought(FeatureEngineeringSignature)
        self.visualization_planner = dspy.ChainOfThought(VisualizationSignature)
        self.statistical_tester = dspy.ChainOfThought(StatisticalTestSignature)
        
        # 优化器
        self.optimizer = BootstrapFewShot(
            metric=self._evaluate_response,
            max_bootstrapped_demos=3,
            max_labeled_demos=5
        )
        
        self._optimize_modules()
    
    def _evaluate_response(self, example, pred, trace=None):
        """评估响应质量"""
        score = 0
        
        # 检查代码是否可执行
        try:
            exec(pred.python_code)
            score += 1
        except:
            pass
        
        # 检查洞察的完整性
        if len(pred.insights) > 50:  # 至少50个字符
            score += 1
        
        return score
    
    def _optimize_modules(self):
        """优化DSPy模块"""
        # 这里可以添加训练数据来优化模块
        pass
    
    def analyze_data(self, task_description, data_context, analysis_type):
        """数据分析任务"""
        try:
            result = self.data_analyzer(
                task_description=task_description,
                data_context=data_context,
                analysis_type=analysis_type
            )
            
            return {
                'analysis_steps': result.analysis_steps,
                'python_code': result.python_code,
                'insights': result.insights
            }
        except Exception as e:
            logger.error(f"数据分析失败: {e}")
            return None
    
    def plan_feature_engineering(self, data_description, columns, 
                                problem_type, target_column=None):
        """特征工程规划"""
        try:
            result = self.feature_engineer(
                data_description=data_description,
                columns=columns,
                target_column=target_column,
                problem_type=problem_type
            )
            
            return {
                'feature_ideas': result.feature_ideas,
                'transformation_code': result.transformation_code,
                'recommendations': result.recommendations
            }
        except Exception as e:
            logger.error(f"特征工程规划失败: {e}")
            return None
    
    def plan_visualization(self, data_info, analysis_goal, plot_types):
        """可视化规划"""
        try:
            result = self.visualization_planner(
                data_info=data_info,
                analysis_goal=analysis_goal,
                plot_types=plot_types
            )
            
            return {
                'visualization_plan': result.visualization_plan,
                'plotly_code': result.plotly_code,
                'insights_from_viz': result.insights_from_viz
            }
        except Exception as e:
            logger.error(f"可视化规划失败: {e}")
            return None
    
    def suggest_statistical_test(self, data_summary, test_objective, hypotheses):
        """统计检验建议"""
        try:
            result = self.statistical_tester(
                data_summary=data_summary,
                test_objective=test_objective,
                hypotheses=hypotheses
            )
            
            return {
                'appropriate_test': result.appropriate_test,
                'test_code': result.test_code,
                'interpretation': result.interpretation
            }
        except Exception as e:
            logger.error(f"统计检验建议失败: {e}")
            return None