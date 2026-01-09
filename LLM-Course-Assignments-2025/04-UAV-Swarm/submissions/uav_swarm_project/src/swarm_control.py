import cosysairsim as airsim
import socket
import json
import time
import threading
import asyncio  # [新增] 引入 asyncio

# --- 配置 ---
HOST = '127.0.0.1'
PORT = 8888
SPEED = 5.0  # 飞行速度 m/s


def fly_drone(drone_name, path_points):
    """
    单个无人机的飞行逻辑函数 (将在独立线程中运行)
    """
    # [关键修复 1] 为当前线程设置一个新的 asyncio 事件循环
    # 解决 RuntimeError: There is no current event loop in thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    print(f"[{drone_name}] 接收到 {len(path_points)} 个航点，准备起飞...")

    # [关键修复 2] 在线程内部创建独立的 AirSim 客户端
    # 不要使用全局共享的 client，每个线程拥有独立的 TCP 连接
    client = airsim.MultirotorClient()
    client.confirmConnection()

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

    # 执行路径规划
    client.moveOnPathAsync(
        airsim_path,
        SPEED,
        timeout_sec=3000,
        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
        yaw_mode=airsim.YawMode(False, 0),
        lookahead=-1,
        adaptive_lookahead=1,
        vehicle_name=drone_name
    ).join()

    print(f"[{drone_name}] 抵达终点，任务完成。")
    # client.landAsync(vehicle_name=drone_name) # 可选：到达后降落

    # 任务结束，断开连接
    client.enableApiControl(False, vehicle_name=drone_name)


def handle_mission(json_data):
    """
    解析 JSON 并启动多线程任务 (带起飞间隔)
    """
    drones_data = json_data.get("drones", {})
    if not drones_data:
        print("[Error] JSON 中没有 drones 数据")
        return

    print(f"[Mission] 收到集群任务，包含 {len(drones_data)} 架无人机")

    threads = []

    # [关键步骤 1] 排序
    # 确保按照 UAV_0, UAV_1, UAV_2 的顺序启动
    # key=... 这一行是为了提取 "UAV_0" 最后的数字 0 进行排序，防止 "UAV_10" 排在 "UAV_2" 前面
    try:
        sorted_drones = sorted(drones_data.items(), key=lambda item: int(item[0].split('_')[-1]))
    except:
        # 如果名字格式不规范，退化为普通字母排序
        sorted_drones = sorted(drones_data.items())

    # [关键步骤 2] 依次启动并等待
    for drone_name, data in sorted_drones:
        path = data.get("path", [])
        if not path: continue

        print(f"[Mission] >>> 启动 {drone_name} 任务...")

        # 创建并启动线程
        t = threading.Thread(target=fly_drone, args=(drone_name, path))
        threads.append(t)
        t.start()

        # [关键修改] 暂停 5 秒再进入下一次循环
        # 这样下一架飞机就会晚 5 秒收到起飞指令
        print(f"[Mission] 间隔等待 5 秒...")
        time.sleep(5)

    # 循环结束后，说明所有飞机都已安排起飞
    # 现在等待所有线程结束 (所有飞机都飞完并悬停)
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