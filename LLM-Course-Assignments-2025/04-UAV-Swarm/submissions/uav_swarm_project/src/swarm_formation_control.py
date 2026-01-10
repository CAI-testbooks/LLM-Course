import cosysairsim as airsim
import socket
import json
import time
import threading
import asyncio
import numpy as np

# --- 全局配置 ---
HOST = '127.0.0.1'
PORT = 8888
SPEED = 3.0  # 建议速度稍微慢一点，保持队形更稳

# [核心] 同步屏障
# 我们稍后根据无人机数量初始化它
# 它的作用是：就像一道闸门，只有凑齐了指定数量的线程，闸门才会打开
sync_barrier = None


# --- [新增] 路径平滑算法 (Catmull-Rom Spline) ---
def catmull_rom_spline(P0, P1, P2, P3, num_points=20):
    """
    计算 P1 到 P2 之间的插值点，受到 P0 和 P3 的张力影响
    """
    t = np.linspace(0, 1, num_points)
    t2 = t * t
    t3 = t2 * t

    # Catmull-Rom 矩阵公式
    # 0.5 * [ (2*P1) + (-P0 + P2)*t + (2*P0 - 5*P1 + 4*P2 - P3)*t2 + (-P0 + 3*P1 - 3*P2 + P3)*t3 ]

    v0 = (P2 - P0) * 0.5
    v1 = (P3 - P1) * 0.5

    # Hermite 形式
    # P(t) = (2t^3 - 3t^2 + 1)P1 + (t^3 - 2t^2 + t)v0 + (-2t^3 + 3t^2)P2 + (t^3 - t^2)v1

    path = (2 * t3 - 3 * t2 + 1)[:, np.newaxis] * P1 + \
           (t3 - 2 * t2 + t)[:, np.newaxis] * v0 + \
           (-2 * t3 + 3 * t2)[:, np.newaxis] * P2 + \
           (t3 - t2)[:, np.newaxis] * v1

    return path


def smooth_path_catmull_rom(raw_points, density=10):
    """
    对原始路径点进行 Catmull-Rom 平滑插值
    raw_points: list of dict {'x':, 'y':, 'z':}
    """
    if len(raw_points) < 2:
        return raw_points

    # 1. 转换为 numpy 数组
    points = np.array([[p['x'], p['y'], p['z']] for p in raw_points])

    # 2. 为了闭合计算，首尾增加辅助点 (简单重复)
    # P_start -> P0 -> P1 ... -> Pn -> P_end
    points = np.vstack([points[0], points, points[-1]])

    smoothed_path = []

    # 3. 逐段插值
    for i in range(len(points) - 3):
        P0, P1, P2, P3 = points[i], points[i + 1], points[i + 2], points[i + 3]

        # 计算 P1 到 P2 这一段的插值
        # 距离越远，插入的点应该越多，这里简单用固定密度
        segment_dist = np.linalg.norm(P2 - P1)
        num_samples = max(5, int(segment_dist * density))  # 每米 density 个点

        segment = catmull_rom_spline(P0, P1, P2, P3, num_samples)

        # 排除第一个点避免重复 (除了整个路径的起点)
        if i > 0:
            smoothed_path.extend(segment[1:])
        else:
            smoothed_path.extend(segment)

    # 转换回字典格式
    result = [{'x': p[0], 'y': p[1], 'z': p[2]} for p in smoothed_path]
    return result


def fly_drone_in_formation(drone_name, path_points):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # --- [新增] 进行路径平滑 ---
        print(f"[{drone_name}] 接收原始路径点: {len(path_points)}")
        # density=2 表示每米插值2个点，将折线变成曲线
        smoothed_points = smooth_path_catmull_rom(path_points, density=2)
        print(f"[{drone_name}] 平滑后路径点: {len(smoothed_points)}")

        client = airsim.MultirotorClient()
        client.confirmConnection()

        # 转换为 AirSim 格式
        airsim_path = []
        for pt in smoothed_points:
            airsim_path.append(airsim.Vector3r(pt['x'], pt['y'], pt['z']))

        client.enableApiControl(True, vehicle_name=drone_name)
        client.armDisarm(True, vehicle_name=drone_name)

        # 垂直起飞到第一个路径点的高度
        if len(airsim_path) > 0:
            first_z = airsim_path[0].z_val
            # 注意 AirSim Z 是负数。如果 first_z 是 -10米，就飞到 -10
            client.takeoffAsync(vehicle_name=drone_name).join()
            client.moveToZAsync(first_z, 2.0, vehicle_name=drone_name).join()
        else:
            client.takeoffAsync(vehicle_name=drone_name).join()

        print(f"[{drone_name}] 等待编队...")
        try:
            sync_barrier.wait(timeout=60)
        except threading.BrokenBarrierError:
            print("超时")
            return

        print(f"[{drone_name}] >>> 出发！")

        # 执行平滑后的路径
        client.moveOnPathAsync(
            airsim_path,
            SPEED,
            timeout_sec=3000,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(False, 0),
            lookahead=-1,  # 自动前视距离
            adaptive_lookahead=0,  # 关闭自适应，因为点很密集，直接跟就行
            vehicle_name=drone_name
        ).join()

        print(f"[{drone_name}] 抵达。")
        client.hoverAsync(vehicle_name=drone_name).join()
        while True: time.sleep(10)

    except Exception as e:
        print(f"[{drone_name}] Error: {e}")


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