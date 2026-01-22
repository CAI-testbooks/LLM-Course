import json
import time
from openai import OpenAI


class SwarmBrain:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")
        self.model = "deepseek-ai/DeepSeek-V3"
        self.history = []  # 记忆库

    def analyze_situation(self, formation_error, obstacle_density, current_shape):
        """
        核心功能：反思与微调
        当底层反馈“队形保持不住了”或“障碍物太多”，LLM介入决策。
        """
        prompt = f"""
        你是一个无人机编队指挥官。当前飞行状态如下：
        - 当前队形: {current_shape}
        - 队形维持误差: {formation_error:.2f} (超过 3.0 表示队形已散乱)
        - 局部障碍物密度: {obstacle_density:.2f} (0-10，越大越危险)

        【任务】
        请反思当前情况，并对编队参数进行微调。

        【可选策略】
        1. "KEEP": 保持现状，一切正常。
        2. "EXPAND": 扩大队形间距（应对障碍物稀疏但分布广的情况）。
        3. "SHRINK": 收缩队形（通过狭窄区域）。
        4. "ROTATE": 旋转队形角度。
        5. "CHANGE_COLUMN": 无法维持三角形，切换为一字纵队。

        请输出 JSON 格式：{{ "thought": "反思过程...", "strategy": "策略代码", "spacing_factor": 1.0 }}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是由大模型驱动的高级飞行控制中枢。"},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            decision = json.loads(response.choices[0].message.content)
            self.history.append(decision)  # 记录反思历史
            return decision
        except Exception as e:
            print(f"LLM Brain Sleep (Error): {e}")
            return {"strategy": "KEEP", "spacing_factor": 1.0}