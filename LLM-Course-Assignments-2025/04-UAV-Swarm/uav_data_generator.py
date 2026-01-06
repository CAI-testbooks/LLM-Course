import numpy as np
import pandas as pd
import airsim
import time


# 生成模拟巡检任务点（10个巡检点，三维坐标）
def generate_task_points(num_points=10):
    np.random.seed(42)
    task_points = np.random.uniform(low=[-20, -20, -20], high=[20, 20, -10], size=(num_points, 3))
    return task_points


# 生成单无人机模拟轨迹
def generate_uav_trajectory(task_points, uav_id, num_steps=100):
    np.random.seed(uav_id)
    trajectory = []
    # 显式指定为浮点数类型（float64）
    current_pos = np.array([uav_id * 5, 0, -20], dtype=np.float64)  # 关键修改
    for step in range(num_steps):
        # 随机向最近的任务点移动
        dists = np.linalg.norm(task_points - current_pos, axis=1)
        target_point = task_points[np.argmin(dists)]
        move_step = (target_point - current_pos) * 0.1  # 步长0.1
        current_pos += move_step + np.random.normal(0, 0.5, 3)  # 加噪声

        trajectory.append({
            "uav_id": uav_id,
            "step": step,
            "x": current_pos[0],
            "y": current_pos[1],
            "z": current_pos[2],
            "velocity": np.linalg.norm(move_step),
            "nearest_task_id": np.argmin(dists)
        })
    return trajectory


# 生成3架无人机的模拟数据集
def generate_sim_dataset():
    task_points = generate_task_points(10)
    all_trajectories = []
    for uav_id in range(3):
        all_trajectories.extend(generate_uav_trajectory(task_points, uav_id))

    # 保存为CSV（用于MARL预训练）
    df = pd.DataFrame(all_trajectories)
    df.to_csv("uav_swarm_sim_data.csv", index=False)
    print("模拟数据集生成完成，保存至uav_swarm_sim_data.csv")


# AirSim实时轨迹采集
def collect_airsim_trajectory(uav_name, duration=30, step=0.5):
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True, uav_name)
    client.armDisarm(True, uav_name)

    trajectory = []
    start_time = time.time()
    while time.time() - start_time < duration:
        # 获取无人机状态
        state = client.getMultirotorState(vehicle_name=uav_name)
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity

        trajectory.append({
            "uav_id": uav_name,
            "time": time.time() - start_time,
            "x": pos.x_val,
            "y": pos.y_val,
            "z": pos.z_val,
            "vx": vel.x_val,
            "vy": vel.y_val,
            "vz": vel.z_val
        })
        time.sleep(step)

    # 保存轨迹
    df = pd.DataFrame(trajectory)
    df.to_csv(f"airsim_{uav_name}_trajectory.csv", index=False)
    print(f"{uav_name}轨迹采集完成")
    return trajectory


# 主函数（可选执行）
if __name__ == "__main__":
    generate_sim_dataset()
    # collect_airsim_trajectory("UAV0")  # 如需采集实时轨迹，取消注释