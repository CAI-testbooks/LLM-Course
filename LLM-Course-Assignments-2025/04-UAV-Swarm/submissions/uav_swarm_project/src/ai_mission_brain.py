import socket
import json
import requests
import math

# --- 你的配置 ---
API_KEY = "sk-pgajauuypprsyzlcctewuurkpctqmnhjrlatbijkbhjjtere"
BASE_URL = "https://api.siliconflow.cn/v1/chat/completions"
HOST = '127.0.0.1'
PORT = 9999


class AITacticalBrain:
    def __init__(self):
        self.headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

    def get_decision(self, scene_data):
        uav_pos = scene_data.get('uav_current_pos', {'x': 0, 'y': 0, 'z': 0})

        system_prompt = """
        你是指挥无人机集群的战术 AI。
        你将收到无人机当前位置和一组目标点（含坐标、Urgency、Deadline）。

        【任务】
        请规划一条包含多个航段（Segment）的巡检路径。
        对于每一个航段（从 A 点飞往 B 点），你必须单独计算飞行速度。

        【速度计算逻辑】
        1. 估算两点间欧几里得距离 Distance (米)。
        2. 计算剩余时间 TimeLeft = Target.Deadline - CurrentTime (假设当前时间为0，即直接用 Deadline)。
        3. 基础速度 V = Distance / TimeLeft。
        4. 修正规则：
           - 速度范围限制在 3.0 m/s 到 20.0 m/s。
           - 如果 Urgency="High"，即使时间充裕，速度也不应低于 10.0 m/s。
           - 如果 Urgency="Low" 且时间充裕，速度保持在 5.0 m/s 以节能。
        5. 必须以 "Airport_Start" 作为最后一个点返航。

        【输出格式】
        Strict JSON only:
        {
            "reasoning": "分析思路...",
            "mission_segments": [
                {
                    "target_id": "Hospital_High",
                    "speed": 18.5,
                    "reason": "Deadline 60s, distance 800m"
                },
                {
                    "target_id": "OilTank_Medium",
                    "speed": 8.0,
                    "reason": "Time sufficient"
                },
                {
                    "target_id": "Airport_Start",
                    "speed": 5.0,
                    "reason": "Return to base"
                }
            ]
        }
        """

        user_prompt = f"战场数据：\n{json.dumps(scene_data, ensure_ascii=False, indent=2)}"

        try:
            print("[AI] Requesting Segmented Plan...")
            payload = {
                "model": "deepseek-ai/DeepSeek-V3",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.1,
                "response_format": {"type": "json_object"}
            }

            response = requests.post(BASE_URL, headers=self.headers, json=payload, timeout=20)
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                return json.loads(content)
        except Exception as e:
            print(f"[AI Error] {e}")
        return None


def start_server():
    brain = AITacticalBrain()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(1)
    print("AI Brain Listening on 9999 (Segmented Mode)...")

    while True:
        try:
            conn, addr = server.accept()
            data = conn.recv(10240).decode('utf-8')
            if not data: continue

            scene_info = json.loads(data)
            decision = brain.get_decision(scene_info)

            if decision:
                # 构造回包，直接把 mission_segments 发给 UE
                response = {
                    "cmd": "segmented_plan",
                    "segments": decision['mission_segments'],
                    "ai_reason": decision['reasoning']
                }
                conn.sendall(json.dumps(response, ensure_ascii=False).encode('utf-8'))
                print(f"[AI] Decision Sent: {len(decision['mission_segments'])} segments")

            conn.close()
        except Exception as e:
            print(f"Server Error: {e}")
            break


if __name__ == "__main__":
    start_server()