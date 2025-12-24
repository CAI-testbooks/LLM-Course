import airsim
import time
import numpy as np
from llm_planner.task_allocator import LLMCommander
from swarm_ctrl.drone_client import SwarmManager



def draw_3d_zone(client, bounds, color, height=-10.0):
    """在 AirSim 中绘制 3D 立体线框"""
    z_floor = 0
    z_ceil = height  # AirSim 中负值代表上方

    # 统一使用 bounds 变量名
    p = [
        airsim.Vector3r(bounds['x_min'], bounds['y_min'], z_floor),  # 0
        airsim.Vector3r(bounds['x_max'], bounds['y_min'], z_floor),  # 1
        airsim.Vector3r(bounds['x_max'], bounds['y_max'], z_floor),  # 2
        airsim.Vector3r(bounds['x_min'], bounds['y_max'], z_floor),  # 3
        airsim.Vector3r(bounds['x_min'], bounds['y_min'], z_ceil),  # 4
        airsim.Vector3r(bounds['x_max'], bounds['y_min'], z_ceil),  # 5
        airsim.Vector3r(bounds['x_max'], bounds['y_max'], z_ceil),  # 6
        airsim.Vector3r(bounds['x_min'], bounds['y_max'], z_ceil)  # 7
    ]

    # 构造 12 条棱边
    lines = [
        p[0], p[1], p[1], p[2], p[2], p[3], p[3], p[0],  # 底面
        p[4], p[5], p[5], p[6], p[6], p[7], p[7], p[4],  # 顶面
        p[0], p[4], p[1], p[5], p[2], p[6], p[3], p[7]  # 四根立柱
    ]

    client.simPlotLineList(lines, color, thickness=5, duration=600, is_persistent=True)


def main():
    # 1. 初始化
    API_KEY = "sk-pgajauuypprsyzlcctewuurkpctqmnhjrlatbijkbhjjtere"
    DRONE_NAMES = ["Drone1", "Drone2", "Drone3"]
    manager = SwarmManager(vehicle_names=DRONE_NAMES)
    commander = LLMCommander(api_key=API_KEY)

    # 2. 获取 LLM 分配并绘制 3D 区域
    total_area = {"x_min": 0, "x_max": 80, "y_min": 0, "y_max": 80}
    assignment = commander.allocate_tasks("按Y轴平分区域，立体搜索", DRONE_NAMES, total_area)

    colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
    for i, task in enumerate(assignment['tasks']):
        draw_3d_zone(manager.client, task['bounds'], colors[i])

    # 3. 集成避障的起飞与中心点前往
    manager.takeoff_all()

    # 目标：每架飞机的中心点坐标
    targets = {}
    for task in assignment['tasks']:
        b = task['bounds']
        targets[task['drone_id']] = np.array([(b['x_min'] + b['x_max']) / 2, (b['y_min'] + b['y_max']) / 2, -5.0])

    print("正在执行避障导航前往中心点...")

    arrival_threshold = 1.5
    active_drones = list(DRONE_NAMES)

    while active_drones:
        all_states = manager.get_all_states()

        for drone_id in active_drones[:]:
            # 当前位置
            curr_pos = np.array([all_states[drone_id]['x'], all_states[drone_id]['y'], all_states[drone_id]['z']])
            target_pos = targets[drone_id]

            # 检查是否到达
            dist = np.linalg.norm(curr_pos[:2] - target_pos[:2])
            if dist < arrival_threshold:
                print(f"{drone_id} 已到达中心并进入悬停模式")
                active_drones.remove(drone_id)
                continue

            # 计算基础引力向量 (方向指向中心)
            direction = target_pos - curr_pos
            direction = direction / np.linalg.norm(direction) * 4.0  # 设定速度为 4m/s

            # --- 避障逻辑介入 ---
            # 1. 雷达避障 (静态障碍物)
            avoid_vector = manager.get_lidar_safe_direction(drone_id)

            # 2. 邻机避障 (动态无人机)
            swarm_vector = manager.avoid_other_drones(drone_id, all_states)

            # 叠加向量
            final_vel = direction + np.array([avoid_vector.x_val, avoid_vector.y_val, 0]) + \
                        np.array([swarm_vector.x_val, swarm_vector.y_val, 0])

            # 执行微步移动
            manager.client.moveToPositionAsync(
                curr_pos[0] + final_vel[0],
                curr_pos[1] + final_vel[1],
                target_pos[2],
                velocity=1, vehicle_name=drone_id
            )

        time.sleep(0.2)  # 5Hz 频率更新

    print("所有无人机已在指定 3D 区域中心安全悬停。")


if __name__ == "__main__":
    main()