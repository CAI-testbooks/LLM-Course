import socket
import json
import time

# 目标服务器配置
SERVER_IP = '127.0.0.1'
SERVER_PORT = 9999


def simulate_ue_request():
    # 1. 构造模拟的场景数据 (这是 UE 将来要发出的格式)
    # 注意坐标单位是 cm (UE标准)
    ue_payload = {
        "timestamp": time.time(),
        "uav_current_pos": {"x": 0.0, "y": 0.0, "z": 200.0},
        "targets": [
            {
                "id": "OilTank_Medium",
                "x": 5000.0,  # 50米处
                "y": 0.0,
                "z": 0.0,
                "urgency": "Medium",
                "deadline": 300  # 5分钟，比较宽裕
            },
            {
                "id": "Hospital_High",
                "x": -8000.0,  # 80米处
                "y": 8000.0,
                "z": 0.0,
                "urgency": "High",
                "deadline": 60  # 1分钟！非常紧急！AI 应该把这个排前面并提高速度
            },
            {
                "id": "ParkingLot_Low",
                "x": 2000.0,
                "y": 2000.0,
                "z": 0.0,
                "urgency": "Low",
                "deadline": 600  # 10分钟
            },
            {
                "id": "Airport_Start",  # 返航点
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "urgency": "Low",
                "deadline": 9999
            }
        ]
    }

    print("--- [Simulator] 正在模拟 UE 发送数据 ---")
    json_str = json.dumps(ue_payload)

    try:
        # 2. 建立连接
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((SERVER_IP, SERVER_PORT))

        # 3. 发送数据
        client.sendall(json_str.encode('utf-8'))
        print("[Simulator] 数据已发送，等待 AI 响应...")

        # 4. 接收响应
        response = client.recv(4096).decode('utf-8')

        print("\n--- [Simulator] 收到 AI 指令 ---")
        # 格式化打印 JSON
        try:
            res_json = json.loads(response)
            print(json.dumps(res_json, indent=4, ensure_ascii=False))

            # 简单的自动验证
            seq = res_json.get('sequence', [])
            speed = res_json.get('speed', 0)

            print("\n--- [自动验证结果] ---")
            if "Hospital_High" in seq and seq[0] == "Hospital_High":
                print("✅ 逻辑正确: 高紧急度目标排在第一位。")
            else:
                print("❌ 逻辑警告: AI 没有将高紧急度目标排在第一位，请检查 Prompt。")

            if speed > 10.0:
                print(f"✅ 速度调整正确: 速度为 {speed} m/s (检测到 Deadline 紧张，已加速)。")
            else:
                print(f"⚠️ 速度较低: {speed} m/s (可能 AI 认为时间足够)。")

        except:
            print(f"原始响应: {response}")

        client.close()

    except ConnectionRefusedError:
        print("❌ 连接失败: 请先运行 ai_mission_brain.py")
    except Exception as e:
        print(f"❌ 发生错误: {e}")


if __name__ == "__main__":
    simulate_ue_request()