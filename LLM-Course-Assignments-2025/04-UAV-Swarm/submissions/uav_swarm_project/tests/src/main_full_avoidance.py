import airsim
import time
import numpy as np
import sys
import os

# 确保导入路径正确
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))
from llm_planner.task_allocator import LLMCommander
from swarm_ctrl.drone_client import SwarmManager


class SwarmIntelligentManager(SwarmManager):
    def get_all_repulsions(self, vehicle_name, all_vehicle_states, safety_radius=5.0):
        """计算全向环境斥力 + 机间斥力"""
        # 1. 环境斥力 (雷达点云)
        lidar_data = self.client.getLidarData(lidar_name="Lidar1", vehicle_name=vehicle_name)
        env_repulsion = np.array([0.0, 0.0, 0.0])

        if len(lidar_data.point_cloud) >= 3:
            points = np.array(lidar_data.point_cloud).reshape(-1, 3)
            distances = np.linalg.norm(points, axis=1)
            danger_points = points[distances < safety_radius]

            for p in danger_points:
                dist = np.linalg.norm(p)
                if dist < 0.2: continue
                # 斥力方向：障碍物指向无人机
                env_repulsion += (-p / dist) * ((safety_radius - dist) / safety_radius)

        # 2. 机间斥力 (防止三机相撞)
        swarm_repulsion = np.array([0.0, 0.0, 0.0])
        my_pos = all_vehicle_states[vehicle_name]

        for other_name, other_pos in all_vehicle_states.items():
            if other_name == vehicle_name: continue

            vec_to_me = np.array([my_pos['x'] - other_pos['x'],
                                  my_pos['y'] - other_pos['y'],
                                  my_pos['z'] - other_pos['z']])
            dist = np.linalg.norm(vec_to_me)
            if dist < 4.0:  # 4米预警距离
                swarm_repulsion += (vec_to_me / dist) * (4.0 - dist)

        return env_repulsion * 2.5 + swarm_repulsion * 4.0


def draw_3d_box(client, bounds, color, height=-8.0):
    """绘制立体任务框"""
    z_f, z_c = 0.0, float(height)
    p = [airsim.Vector3r(float(bounds['x_min']), float(bounds['y_min']), z_f),
         airsim.Vector3r(float(bounds['x_max']), float(bounds['y_min']), z_f),
         airsim.Vector3r(float(bounds['x_max']), float(bounds['y_max']), z_f),
         airsim.Vector3r(float(bounds['x_min']), float(bounds['y_max']), z_f),
         airsim.Vector3r(float(bounds['x_min']), float(bounds['y_min']), z_c),
         airsim.Vector3r(float(bounds['x_max']), float(bounds['y_min']), z_c),
         airsim.Vector3r(float(bounds['x_max']), float(bounds['y_max']), z_c),
         airsim.Vector3r(float(bounds['x_min']), float(bounds['y_max']), z_c)]
    lines = [p[0], p[1], p[1], p[2], p[2], p[3], p[3], p[0], p[4], p[5], p[5], p[6], p[6], p[7], p[7], p[4],
             p[0], p[4], p[1], p[5], p[2], p[6], p[3], p[7]]
    client.simPlotLineList(lines, color, thickness=3, duration=600, is_persistent=True)


def main():
    API_KEY = "sk-pgajauuypprsyzlcctewuurkpctqmnhjrlatbijkbhjjtere"
    DRONE_NAMES = ["Drone1", "Drone2", "Drone3"]
    manager = SwarmIntelligentManager(vehicle_names=DRONE_NAMES)
    commander = LLMCommander(api_key=API_KEY)

    # 1. LLM 区域规划与 3D 绘图
    total_area = {"x_min": 0, "x_max": 80, "y_min": 0, "y_max": 80}
    mission = commander.allocate_tasks("将80x80区域分成三块，用于协同搜索", DRONE_NAMES, total_area)

    colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
    targets = {}
    for i, task in enumerate(mission['tasks']):
        draw_3d_box(manager.client, task['bounds'], colors[i])
        b = task['bounds']
        targets[task['drone_id']] = np.array([(b['x_min'] + b['x_max']) / 2.0, (b['y_min'] + b['y_max']) / 2.0, -10.0])

    # 2. 同步起飞
    manager.takeoff_all()

    # 3. 协同导航循环
    active_drones = list(DRONE_NAMES)
    stuck_counters = {name: 0 for name in DRONE_NAMES}

    print("三机协同避障飞行中...")
    while active_drones:
        all_states = manager.get_all_states()  # 获取所有飞机位置

        for name in active_drones[:]:
            curr_pos = np.array([all_states[name]['x'], all_states[name]['y'], all_states[name]['z']])
            target = targets[name]

            dist_to_goal = np.linalg.norm(curr_pos - target)
            if dist_to_goal < 1.5:
                print(f"{name} 到达指定空域中心")
                active_drones.remove(name)
                continue

            # 向量融合：引力 + 环境斥力 + 机间斥力
            dir_vec = (target - curr_pos) / np.linalg.norm(target - curr_pos) * 3.0
            avoid_vec = manager.get_all_repulsions(name, all_states)
            final_vec = dir_vec + avoid_vec

            # 执行移动 (修复 NumPy 类型)
            manager.client.moveToPositionAsync(
                float(curr_pos[0] + final_vec[0]),
                float(curr_pos[1] + final_vec[1]),
                float(curr_pos[2] + final_vec[2]),
                velocity=1, vehicle_name=name
            )

        time.sleep(0.1)


if __name__ == "__main__":
    main()