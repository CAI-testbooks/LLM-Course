import cosysairsim as airsim
import socket
import json
import time
import threading
import asyncio
import numpy as np  # 需要: pip install numpy

# --- 全局配置 ---
HOST = '127.0.0.1'
PORT = 8888
DEFAULT_SPEED = 5.0  # 如果 JSON 里没写速度，就用这个默认值
sync_barrier = None


# --- [保留] 路径平滑算法 (Catmull-Rom Spline) ---
def catmull_rom_spline(P0, P1, P2, P3, num_points=20):
    t = np.linspace(0, 1, num_points)
    t2 = t * t
    t3 = t2 * t

    v0 = (P2 - P0) * 0.5
    v1 = (P3 - P1) * 0.5

    path = (2 * t3 - 3 * t2 + 1)[:, np.newaxis] * P1 + \
           (t3 - 2 * t2 + t)[:, np.newaxis] * v0 + \
           (-2 * t3 + 3 * t2)[:, np.newaxis] * P2 + \
           (t3 - t2)[:, np.newaxis] * v1

    return path


def smooth_path_catmull_rom(raw_points, density=5):
    """
    对原始路径点进行 Catmull-Rom 平滑插值
    density: 每米插值的点数
    """
    if len(raw_points) < 2:
        return raw_points

    points = np.array([[p['x'], p['y'], p['z']] for p in raw_points])
    # 首尾增加辅助点
    points = np.vstack([points[0], points, points[-1]])

    smoothed_path = []

    for i in range(len(points) - 3):
        P0, P1, P2, P3 = points[i], points[i + 1], points[i + 2], points[i + 3]
        segment_dist = np.linalg.norm(P2 - P1)
        num_samples = max(5, int(segment_dist * density))

        segment = catmull_rom_spline(P0, P1, P2, P3, num_samples)

        if i > 0:
            smoothed_path.extend(segment[1:])
        else:
            smoothed_path.extend(segment)

    return [{'x': p[0], 'y': p[1], 'z': p[2]} for p in smoothed_path]


# --- [修改] 增加 mission_speed 参数 ---
# 修改飞行函数，支持分段执行
def fly_drone_segments(drone_name, segments_data):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        client = airsim.MultirotorClient()
        client.confirmConnection()
        client.enableApiControl(True, vehicle_name=drone_name)
        client.armDisarm(True, vehicle_name=drone_name)

        # 1. 初始起飞 (基于第一段的第一个点，或者原地)
        client.takeoffAsync(vehicle_name=drone_name).join()

        # 2. 循环执行每一段 (Segment Loop)
        for i, seg in enumerate(segments_data):
            raw_path = seg.get('path', [])
            speed = seg.get('speed', 5.0)

            if not raw_path: continue

            print(f"[{drone_name}] 执行第 {i + 1} 段任务 | 速度: {speed} m/s | 节点数: {len(raw_path)}")

            # 路径平滑
            smoothed = smooth_path_catmull_rom(raw_path, density=2)

            # 转 AirSim 格式
            airsim_path = [airsim.Vector3r(p['x'], p['y'], p['z']) for p in smoothed]

            # 第一段需要先飞到起点高度
            if i == 0 and len(airsim_path) > 0:
                client.moveToZAsync(airsim_path[0].z_val, 3.0, vehicle_name=drone_name).join()

                # 等待编队集结 (仅在第一段开始前)
                print(f"[{drone_name}] 等待集结...")
                try:
                    if sync_barrier: sync_barrier.wait(timeout=60)
                except:
                    pass

            # 执行飞行
            client.moveOnPathAsync(
                airsim_path,
                speed,  # 这里的速度是这一段独有的！
                timeout_sec=3000,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(False, 0),
                lookahead=-1, adaptive_lookahead=0,
                vehicle_name=drone_name
            ).join()

            # 每段结束后稍微悬停一下，或者直接衔接下一段
            # print(f"[{drone_name}] 第 {i+1} 段完成")

        print(f"[{drone_name}] 所有任务链执行完毕，悬停。")
        client.hoverAsync(vehicle_name=drone_name).join()
        while True: time.sleep(10)

    except Exception as e:
        print(f"[{drone_name}] Error: {e}")


def handle_mission(json_data):
    global sync_barrier

    drones_data = json_data.get("drones", {})
    if not drones_data: return

    # 初始化 barrier
    sync_barrier = threading.Barrier(len(drones_data))
    threads = []

    print(f"\n[Mission] 收到分段任务，开始调度...")

    for drone_name, data in drones_data.items():
        # 这里 data 里应该有 "segments"
        segments = data.get("segments", [])

        # 兼容旧代码：如果只有 path，把它包装成一个 segment
        if "path" in data:
            speed = json_data.get("speed", 5.0)
            segments = [{"path": data["path"], "speed": speed}]

        if not segments: continue

        t = threading.Thread(target=fly_drone_segments, args=(drone_name, segments))
        t.daemon = True
        threads.append(t)
        t.start()

    for t in threads: t.join()


def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        server.bind((HOST, PORT))
        server.listen(1)
        print(f"==========================================")
        print(f"[AirSim Control Server] 监听端口 {PORT}")
        print(f"支持动态速度解析 (JSON key: 'speed')")
        print(f"==========================================")

        while True:
            conn, addr = server.accept()
            print(f"[Server] 连接来自: {addr}")

            try:
                # 接收大数据包 (循环接收直到无数据)
                # 因为路径点可能很多，4096可能不够
                buffer = b""
                while True:
                    chunk = conn.recv(40960)  # 加大单次接收缓冲
                    if not chunk: break
                    buffer += chunk
                    # 简单的判断：如果 JSON 结尾是 '}' 且花括号成对，可能就收完了
                    # 实际工程建议加包头长度，这里简单处理
                    if len(buffer) > 0 and buffer.strip().endswith(b'}'):
                        # 尝试解析，如果成功就跳出接收循环
                        try:
                            json.loads(buffer.decode('utf-8'))
                            break
                        except:
                            pass  # 还没收完，继续收

                data_str = buffer.decode('utf-8')
                if not data_str:
                    conn.close()
                    continue

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