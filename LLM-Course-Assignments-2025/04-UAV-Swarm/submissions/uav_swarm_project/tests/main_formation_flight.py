import airsim
import time
import numpy as np
from swarm_brain import SwarmBrain
from formation import TriangleFormation
from lidar_pilot import get_combined_force


def main():
    # 1. 初始化
    client = airsim.MultirotorClient()
    client.confirmConnection()
    drones = ["Drone1", "Drone2", "Drone3"]

    for d in drones:
        client.enableApiControl(True, d)
        client.armDisarm(True, d)
        client.takeoffAsync(d).join()

    # 2. 定义任务
    brain = SwarmBrain(api_key="sk-pgajauuypprsyzlcctewuurkpctqmnhjrlatbijkbhjjtere")
    formation = TriangleFormation(spacing=6.0)

    # 定义编队整体的移动轨迹 (虚拟领航者的路径)
    # 假设从 (0,0) 飞到 (80, 0)
    start_point = np.array([0.0, 0.0, -5.0])
    end_point = np.array([80.0, 0.0, -15.0])
    current_virtual_leader = start_point.copy()

    print(">>> 编队任务开始：三角形编队 -> 目标点 (80,0)")

    # 3. 控制循环
    step = 0
    while np.linalg.norm(current_virtual_leader - end_point) > 2.0:
        step += 1

        # A. 虚拟领航者向前移动 (速度 2m/s)
        dir_vec = (end_point - current_virtual_leader)
        dir_vec = dir_vec / np.linalg.norm(dir_vec)
        current_virtual_leader += dir_vec * 0.2  # 0.1s update * 2m/s

        # 获取所有飞机当前位置
        all_states = {}
        for d in drones:
            pos = client.getMultirotorState(d).kinematics_estimated.position
            all_states[d] = np.array([pos.x_val, pos.y_val, pos.z_val])

        # 收集数据供 LLM 反思
        total_formation_error = 0
        max_obstacle_force = 0

        # B. 为每架飞机计算控制指令
        for d in drones:
            # 1. 计算理论位置 (Formation)
            target_pos = formation.get_target_position(current_virtual_leader, d)

            # 计算队形误差 (用于向 LLM 汇报)
            error = np.linalg.norm(all_states[d] - target_pos)
            total_formation_error += error

            # 2. 计算合力 (Reactive Avoidance)
            velocity_vec, obs_force = get_combined_force(client, d, target_pos, all_states[d], all_states)
            max_obstacle_force = max(max_obstacle_force, obs_force)

            # 3. 执行控制
            client.moveByVelocityAsync(
                float(velocity_vec[0]), float(velocity_vec[1]), float(velocity_vec[2]),
                0.1, vehicle_name=d
            )

        # C. LLM 介入判断 (每 20 帧，约2秒一次，避免调用太频繁)
        if step % 20 == 0:
            print(f"[监控] 队形误差: {total_formation_error:.1f}, 障碍强度: {max_obstacle_force:.1f}")

            # 触发条件：如果队形散了(误差大) 或者 障碍物极多
            if total_formation_error > 8.0 or max_obstacle_force > 5.0:
                print(">>> ⚠️ 状态异常，请求大模型反思...")
                decision = brain.analyze_situation(total_formation_error, max_obstacle_force, formation.shape_type)

                print(f"LLM 反思: {decision.get('thought')}")
                print(f"LLM 决策: {decision.get('strategy')}")

                # 执行 LLM 的微调指令
                if decision.get('strategy') == "EXPAND":
                    formation.update_spacing(1.2)  # 扩大间距
                elif decision.get('strategy') == "SHRINK":
                    formation.update_spacing(0.8)  # 收缩
                # 这里可以扩展更多逻辑...

        time.sleep(0.1)

    print("任务完成！")
    for d in drones:
        client.landAsync(d)


if __name__ == "__main__":
    main()