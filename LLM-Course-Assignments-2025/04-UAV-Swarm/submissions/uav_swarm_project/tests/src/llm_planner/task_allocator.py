import json

import httpx
from openai import OpenAI

# 环境边界配置 (根据 Floor 缩放 8.0 推算: 10m * 8 = 80m)
# 中心在 (0,0)，半径 40m。留 5m 缓冲，半径设为 35m。
ENV_CONFIG = {
    "x_min": -35.0,
    "x_max": 35.0,
    "y_min": -35.0,
    "y_max": 35.0,
    "z_fly": -5.0  # 飞行高度
}

# 提示词补充信息
ENV_PROMPT_TEXT = f"""
当前飞行空域限制为一个 80x80 米的正方形区域。
有效任务坐标范围：X [{ENV_CONFIG['x_min']}, {ENV_CONFIG['x_max']}], Y [{ENV_CONFIG['y_min']}, {ENV_CONFIG['y_max']}]。
请确保所有航点都在此坐标范围内，切勿越界。
"""


class LLMCommander:
    def __init__(self, api_key):
        # 硅基流动的 API 基础地址
        #sk-pgajauuypprsyzlcctewuurkpctqmnhjrlatbijkbhjjtere
        http_client = httpx.Client(
            proxy=None,  # 强制不使用代理
            transport=httpx.HTTPTransport(retries=3),
            verify=False  # 如果依然报 SSL 错误，可以临时设为 False（不推荐长期使用）
        )
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.siliconflow.cn/v1",
            http_client = http_client  # 将自定义客户端传入
        )
        # 推荐使用 Qwen2.5-7B-Instruct 或 DeepSeek-V3，性价比极高
        self.model = "deepseek-ai/DeepSeek-V3"

    def allocate_tasks(self, user_input, drones, total_area):
        system_prompt = """
        你是一个无人机集群指挥官。你的任务是将一个大矩形区域切分成 N 个子矩形，分配给 N 架无人机。
        必须输出严格的 JSON 格式，不得包含任何文字说明。
        JSON 结构示例：
        {
          "tasks": [
            {"drone_id": "Drone1", "bounds": {"x_min": 0, "x_max": 30, "y_min": 0, "y_max": 60}},
            ...
          ]
        }
        """

        user_content = f"指令: {user_input}\n无人机列表: {drones}\n总区域边界: {total_area}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)

    def replan_target(self, drone_id, old_bounds, obstacle_pos):
        """当原中心点不可用时，请求 LLM 在剩余安全区域寻找新点"""
        prompt = f"""
        无人机 {drone_id} 原定区域中心在障碍物附近({obstacle_pos})。
        请在原区域 {old_bounds} 内重新选择一个安全中心点。
        输出 JSON: {{"new_center": [x, y, z]}}
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)