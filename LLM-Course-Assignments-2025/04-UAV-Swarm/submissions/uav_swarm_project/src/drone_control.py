import cosysairsim as airsim
import socket
import json
import time
import os

# --- 配置 ---
HOST = '127.0.0.1'
PORT = 8888
SPEED = 5.0
JSON_FILE = "flight_plan.json"


def connect_to_airsim():
    print("[AirSim] 连接中...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    print("[AirSim] 已连接且解锁")
    return client


def receive_and_save_data(conn):
    """
    循环接收数据直到连接关闭，并保存为 JSON
    """
    buffer = b""
    print("[Network] 开始接收数据...")
    try:
        while True:
            # 每次读取 4096 字节
            chunk = conn.recv(4096)
            if not chunk:
                break  # 连接关闭，传输完成
            buffer += chunk

        data_str = buffer.decode('utf-8')

        # 尝试解析 JSON 确保数据完整
        json_data = json.loads(data_str)

        # 保存到文件
        with open(JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4)

        print(f"[System] 数据已保存至 {JSON_FILE} (大小: {len(buffer)} 字节)")
        return True

    except json.JSONDecodeError as e:
        print(f"[Error] JSON 解析失败 (数据可能不完整): {e}")
        return False
    except Exception as e:
        print(f"[Error] 接收出错: {e}")
        return False


def execute_flight_plan(client):
    """
    读取 JSON 并执行飞行
    """
    if not os.path.exists(JSON_FILE):
        print("[Error] 找不到飞行计划文件")
        return

    print(f"[System] 读取飞行计划: {JSON_FILE}")
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        plan = json.load(f)

    path_points = plan.get("path", [])
    if not path_points:
        print("[Warning] 路径为空")
        return

    print(f"[Flight] 解析到 {len(path_points)} 个航点")

    # 1. 构造 AirSim 路径数组
    airsim_path = []
    for pt in path_points:
        airsim_path.append(airsim.Vector3r(pt['x'], pt['y'], pt['z']))

    # 2. (可选) 瞬移到起点
    # 如果你想让飞机瞬移到路径第一个点，取消下面注释
    start_pt = airsim_path[0]
    start_pose = airsim.Pose(start_pt, airsim.Quaternionr(0,0,0,1))
    print(f"[Flight] 瞬移到起点: {start_pt}")
    client.simSetVehiclePose(start_pose, True)
    time.sleep(1) # 等一下物理引擎

    # 3. 起飞
    print("[Flight] 起飞...")
    client.takeoffAsync().join()

    # 4. 执行路径
    print(f"[Flight] 开始沿路径飞行 (速度: {SPEED} m/s)...")

    # moveOnPathAsync 相比 moveToPosition 更加平滑，适合连续路径
    # drivetrain=MaxDegreeOfFreedom 允许机头自由转动 (或者根据路径切线转动)
    client.moveOnPathAsync(
        airsim_path,
        SPEED,
        timeout_sec=3000,
        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
        yaw_mode=airsim.YawMode(False, 0),
        lookahead=-1,
        adaptive_lookahead=1
    ).join()  # .join() 会等待飞行完成

    print("[Flight] 任务完成，悬停中。")


def start_server():
    # 先连上 AirSim，确保环境没问题
    client = connect_to_airsim()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(1)
    print(f"[Network] 等待 UE 连接 ({HOST}:{PORT})...")

    while True:
        conn, addr = server.accept()
        print(f"[Network] UE 已连接: {addr}")

        # 1. 接收并保存
        success = receive_and_save_data(conn)
        conn.close()  # 关掉连接

        # 2. 如果成功，执行飞行
        if success:
            execute_flight_plan(client)

        print("-" * 30)
        print("等待下一次任务...")


if __name__ == "__main__":
    start_server()