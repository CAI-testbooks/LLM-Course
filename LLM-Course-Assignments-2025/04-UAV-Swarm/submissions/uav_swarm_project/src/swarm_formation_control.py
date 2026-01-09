import cosysairsim as airsim
import socket
import json
import time
import threading
import asyncio

# --- 全局配置 ---
HOST = '127.0.0.1'
PORT = 8888
SPEED = 3.0  # 建议速度稍微慢一点，保持队形更稳

# [核心] 同步屏障
# 我们稍后根据无人机数量初始化它
# 它的作用是：就像一道闸门，只有凑齐了指定数量的线程，闸门才会打开
sync_barrier = None


def fly_drone_in_formation(drone_name, path_points):
    """
    编队飞行逻辑：起飞 -> 等待队友 -> 同时出发
    """
    try:
        # 1. 线程环境初始化
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # 2. 建立独立连接
        client = airsim.MultirotorClient()
        client.confirmConnection()

        # 转换坐标
        airsim_path = []
        for pt in path_points:
            airsim_path.append(airsim.Vector3r(pt['x'], pt['y'], pt['z']))

        print(f"[{drone_name}] 准备就绪，路径节点数: {len(airsim_path)}")

        # 3. 解锁并起飞
        client.enableApiControl(True, vehicle_name=drone_name)
        client.armDisarm(True, vehicle_name=drone_name)

        # 起飞 (各自起飞)
        client.takeoffAsync(vehicle_name=drone_name).join()

        # 爬升到安全高度 (防止有的起飞没到位)
        client.moveToZAsync(-3.0, 1.0, vehicle_name=drone_name).join()

        print(f"[{drone_name}] 已起飞，正在空中悬停等待队友...")

        # =========================================================
        # [关键时刻] 同步等待
        # 线程运行到这里会被阻塞，直到 4 个线程都运行到这一行
        # =========================================================
        try:
            sync_barrier.wait(timeout=60)  # 最多等60秒
        except threading.BrokenBarrierError:
            print(f"[{drone_name}] 等待超时，队形解散！")
            return

        print(f"[{drone_name}] >>> 编队集结完毕，同时出发！ <<<")

        # 4. 同时执行路径
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

        # 5. 抵达终点
        print(f"[{drone_name}] 抵达终点，保持悬停。")
        client.hoverAsync(vehicle_name=drone_name).join()

        # 保持连接
        while True:
            time.sleep(10)

    except Exception as e:
        print(f"[{drone_name}] 异常: {e}")


def handle_mission(json_data):
    global sync_barrier

    drones_data = json_data.get("drones", {})
    if not drones_data:
        return

    drone_count = len(drones_data)
    print(f"\n[Mission] 收到编队任务，无人机数量: {drone_count}")

    # [初始化屏障] 需要等待 drone_count 个线程
    sync_barrier = threading.Barrier(drone_count)

    threads = []

    # 启动所有线程
    # 注意：这里不需要排序了，因为反正要等齐了才飞，顺序不重要
    for drone_name, data in drones_data.items():
        path = data.get("path", [])
        if not path: continue

        t = threading.Thread(target=fly_drone_in_formation, args=(drone_name, path))
        t.daemon = True
        threads.append(t)
        t.start()
        print(f"[Mission] {drone_name} 启动自检程序...")

    # 等待所有线程结束
    for t in threads:
        t.join()


def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        server.bind((HOST, PORT))
        server.listen(1)
        print(f"==========================================")
        print(f"[Formation Server] 编队控制服务已启动")
        print(f"等待 UE 发送指令...")
        print(f"==========================================")

        while True:
            conn, addr = server.accept()
            print(f"[Server] 连接来自: {addr}")

            try:
                buffer = b""
                while True:
                    chunk = conn.recv(4096)
                    if not chunk: break
                    buffer += chunk

                data_str = buffer.decode('utf-8')
                if not data_str: continue

                mission_plan = json.loads(data_str)
                handle_mission(mission_plan)

            except Exception as e:
                print(f"[Error] {e}")
            finally:
                conn.close()

    except Exception as e:
        print(f"[Fatal] {e}")
    finally:
        server.close()


if __name__ == "__main__":
    start_server()