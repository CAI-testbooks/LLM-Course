import cosysairsim as airsim
import socket
import json
import time
import threading

# --- 配置 ---
HOST = '127.0.0.1'
PORT = 8888
SPEED = 3.0  # 飞行速度 m/s


def fly_drone(client, drone_name, path_points):
    """
    单个无人机的飞行逻辑函数 (将在独立线程中运行)
    """
    print(f"[{drone_name}] 接收到 {len(path_points)} 个航点，准备起飞...")

    # 1. 构造 AirSim 路径
    airsim_path = []
    for pt in path_points:
        airsim_path.append(airsim.Vector3r(pt['x'], pt['y'], pt['z']))

    # 2. 确保解锁并起飞
    client.enableApiControl(True, vehicle_name=drone_name)
    client.armDisarm(True, vehicle_name=drone_name)

    # 异步起飞，等待完成
    client.takeoffAsync(vehicle_name=drone_name).join()

    # 3. 执行飞行
    print(f"[{drone_name}] 开始飞行...")

    # 注意：这里我们使用 join() 等待飞行完成，因为这个函数本身就是在独立线程里跑的
    # 这样不会阻塞其他无人机
    client.moveOnPathAsync(
        airsim_path,
        SPEED,
        timeout_sec=3000,
        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
        yaw_mode=airsim.YawMode(False, 0),
        lookahead=-1,
        adaptive_lookahead=1,
        vehicle_name=drone_name  # 关键：指定控制哪架飞机
    ).join()

    print(f"[{drone_name}] 抵达终点，任务完成。")
    # client.landAsync(vehicle_name=drone_name) # 可选：到达后降落


def handle_mission(json_data):
    """
    解析 JSON 并启动多线程任务
    """
    drones_data = json_data.get("drones", {})
    if not drones_data:
        print("[Error] JSON 中没有 drones 数据")
        return

    print(f"[Mission] 收到集群任务，包含 {len(drones_data)} 架无人机")

    # 创建 AirSim 客户端 (主线程)
    # 注意：AirSim Client 是线程安全的，可以在多线程中共享，
    # 但为了保险，我们在每个线程里单独控制 vehicle_name
    client = airsim.MultirotorClient()
    client.confirmConnection()

    threads = []

    # 遍历每架无人机，启动一个线程去飞
    for drone_name, data in drones_data.items():
        path = data.get("path", [])
        if not path: continue

        # 创建线程
        t = threading.Thread(target=fly_drone, args=(client, drone_name, path))
        threads.append(t)
        t.start()

    # 等待所有飞机飞完
    for t in threads:
        t.join()

    print("[Mission] 所有无人机任务均已结束。")


def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(1)
    print(f"[Server] 等待 UE 连接 ({HOST}:{PORT})...")

    while True:
        conn, addr = server.accept()
        print(f"[Server] UE 已连接: {addr}")

        try:
            # 接收数据
            buffer = b""
            while True:
                chunk = conn.recv(4096)
                if not chunk: break
                buffer += chunk

            data_str = buffer.decode('utf-8')
            if not data_str: continue

            # 解析 JSON
            mission_plan = json.loads(data_str)

            # 保存备份
            with open("swarm_plan.json", "w") as f:
                json.dump(mission_plan, f, indent=4)

            # 执行任务
            handle_mission(mission_plan)

        except Exception as e:
            print(f"[Error] 处理任务失败: {e}")
        finally:
            conn.close()
            print("[Server] 等待下一次连接...")


if __name__ == "__main__":
    start_server()