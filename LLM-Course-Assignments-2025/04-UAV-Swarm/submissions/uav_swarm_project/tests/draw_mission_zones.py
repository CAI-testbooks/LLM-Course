import airsim
import sys
import os
import time

# 确保能找到 src 目录下的模块
# 动态获取当前脚本所在位置的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 定位到项目根目录 (uav_swarm_project)
project_root = os.path.abspath(os.path.join(current_dir, '..'))
# 定位到 src 目录
src_path = os.path.join(project_root, 'src')

if src_path not in sys.path:
    sys.path.append(src_path)

from llm_planner.task_allocator import LLMCommander
from swarm_ctrl.drone_client import SwarmManager


def main():
    # --- 1. 配置区 ---
    API_KEY = "sk-pgajauuypprsyzlcctewuurkpctqmnhjrlatbijkbhjjtere"
    DRONE_NAMES = ["Drone1", "Drone2", "Drone3"]
    TOTAL_AREA = {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 100}

    # 实验指令：你可以修改这里测试 LLM 的理解力
    USER_CMD = "Drone1性能最好，给它分一半区域，剩下的给另外两架平均分。"

    # --- 2. 初始化 ---
    commander = LLMCommander(api_key=API_KEY)
    manager = SwarmManager(vehicle_names=DRONE_NAMES)
    client = manager.client  # 获取底层的 airsim client

    print(f"正在请求 LLM 分配任务: {USER_CMD}")
    assignment = commander.allocate_tasks(USER_CMD, DRONE_NAMES, TOTAL_AREA)

    # --- 3. AirSim 3D 可视化绘制 (1.8.1 兼容版) ---
    print("正在使用 simPlotLineList 绘制任务区域...")

    # 构造线条列表
    # simPlotLineList 接收的是一个 Vector3r 列表，每两个点组成一条线
    all_lines = []
    all_colors = []

    colors = [
        [1.0, 0.0, 0.0, 1.0],  # Drone1: 红
        [0.0, 1.0, 0.0, 1.0],  # Drone2: 绿
        [0.0, 0.0, 1.0, 1.0]  # Drone3: 蓝
    ]

    for i, task in enumerate(assignment['tasks']):
        b = task['bounds']
        color = airsim.Vector3r(colors[i % len(colors)][0],
                                colors[i % len(colors)][1],
                                colors[i % len(colors)][2])

        # 定义矩形的四个顶点
        p1 = airsim.Vector3r(b['x_min'], b['y_min'], 0)
        p2 = airsim.Vector3r(b['x_max'], b['y_min'], 0)
        p3 = airsim.Vector3r(b['x_max'], b['y_max'], 0)
        p4 = airsim.Vector3r(b['x_min'], b['y_max'], 0)

        # 构造线段对: p1-p2, p2-p3, p3-p4, p4-p1
        rect_points = [p1, p2, p2, p3, p3, p4, p4, p1]
        all_lines.extend(rect_points)

        # 为每条线段分配颜色 (每条线由两个点组成)
        for _ in range(4):
            all_colors.append(colors[i % len(colors)])

    # 调用 1.8.1 核心绘图 API
    # simPlotLineList(points, color_rgba, thickness, duration, is_persistent)
    try:
        # 注意：1.8.1 有些子版本不支持 list 传 color，我们先尝试为整组线画一种颜色，
        # 或者循环调用单条线绘制
        for i in range(0, len(all_lines), 2):
            point_pair = [all_lines[i], all_lines[i + 1]]
            current_color = all_colors[i // 2]
            client.simPlotLineList(point_pair, current_color, thickness=10.0, duration=300.0, is_persistent=True)
        print("线条绘制指令已发送。")
    except Exception as e:
        print(f"绘图失败: {e}。尝试使用 simPrintText 标记位置。")

    # --- 4. 无人机同步动作验证 ---
    print("起飞并前往区域中心...")
    manager.takeoff_all()

    # 让每架无人机飞往其分配区域的中心点 (高度 -5m)
    move_tasks = {}
    for task in assignment['tasks']:
        d_id = task['drone_id']
        b = task['bounds']
        target_pos = [(b['x_min'] + b['x_max']) / 2, (b['y_min'] + b['y_max']) / 2, -5]
        move_tasks[d_id] = target_pos

    futures = manager.move_to_positions_async(move_tasks, velocity=5)
    for f in futures:
        f.join()

    print("所有无人机已到达 LLM 指定区域中心，请在 AirSim 中查看可视化线条。")
    time.sleep(10)
    # manager.land_all() # 测试时可以先不降落，方便观察


if __name__ == "__main__":
    main()