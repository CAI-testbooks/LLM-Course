import airsim
import numpy as np
import json
from sklearn.cluster import DBSCAN
import time


def lidar_scan_and_map():
    print("Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()

    # 1. 起飞一点点，以便雷达能扫到地面以上的物体
    print("Taking off for better scan view...")
    client.enableApiControl(True, "Drone1")
    client.armDisarm(True, "Drone1")
    # 飞到 5米高度 (AirSim中 负数是向上)
    client.takeoffAsync("Drone1").join()
    client.moveToPositionAsync(0, 0, -5, 2, "Drone1").join()

    # 等待稳定
    time.sleep(2)

    print("Scanning environment with Lidar...")

    # 2. 获取雷达数据
    lidar_data = client.getLidarData(lidar_name="Lidar1", vehicle_name="Drone1")

    if len(lidar_data.point_cloud) < 3:
        print("No lidar points received!")
        return

    # 3. 解析点云 (x, y, z)
    # AirSim 返回的是一维 list，需要 reshape 成 (N, 3)
    points = np.array(lidar_data.point_cloud).reshape(-1, 3)

    # 获取无人机当前位置（用于将相对坐标转为绝对坐标）
    state = client.getMultirotorState(vehicle_name="Drone1")
    drone_pos = state.kinematics_estimated.position
    # 修正坐标系：AirSim Lidar 数据是相对于 Drone 的 Body Frame 的
    # 简单起见，我们假设无人机朝向未变（初始状态），直接加上无人机坐标
    # 如果有旋转，需要乘以旋转矩阵。这里做简单处理用于获取静态环境。
    global_points = points + np.array([drone_pos.x_val, drone_pos.y_val, drone_pos.z_val])

    # 4. 数据清洗：去除地面点
    # 假设地面在 z=0 附近（AirSim坐标系下 Z正方向是下）。
    # 如果无人机在 -5米，地面相对无人机就是 +5米。
    # 我们过滤掉 Z > 2 的点（即比无人机低2米以上的点，视为地面）
    obstacle_points = global_points[global_points[:, 2] < 2.0]

    print(f"Valid obstacle points after filtering ground: {len(obstacle_points)}")

    if len(obstacle_points) == 0:
        print("No obstacles detected (only ground found).")
        return

    # 5. 使用 DBSCAN 聚类算法识别独立物体
    # eps=3.0: 同一个物体的点之间距离不超过3米
    # min_samples=10: 至少10个点才算一个物体（过滤噪点）
    clustering = DBSCAN(eps=3.0, min_samples=10).fit(obstacle_points)
    labels = clustering.labels_

    # 提取聚类中心
    unique_labels = set(labels)
    found_obstacles = []

    print(f"Found {len(unique_labels) - (1 if -1 in labels else 0)} unique clusters.")

    for k in unique_labels:
        if k == -1: continue  # -1 是噪声点

        # 提取属于这一类的所有点
        class_member_mask = (labels == k)
        xy_points = obstacle_points[class_member_mask]

        # 计算中心点 (Centroid)
        center = np.mean(xy_points, axis=0)

        # 估算大小 (Bounding Box)
        min_bound = np.min(xy_points, axis=0)
        max_bound = np.max(xy_points, axis=0)
        size = max_bound - min_bound

        # 记录数据
        found_obstacles.append({
            "name": f"Detected_Obstacle_{k}",
            "pos": [round(center[0], 2), round(center[1], 2), round(center[2], 2)],
            "size": [round(size[0], 2), round(size[1], 2), round(size[2], 2)],
            "dist": round(np.linalg.norm(center[:2]), 2)  # 距离原点的距离
        })
        print(f" -> Object {k}: Pos={found_obstacles[-1]['pos']}, Size={found_obstacles[-1]['size']}")

    # 6. 保存为 JSON
    with open("ue5_env_scan.json", "w", encoding='utf-8') as f:
        json.dump(found_obstacles, f, indent=4)

    print("Environment scan saved to ue5_env_scan.json based on Lidar data.")

    # 降落
    client.landAsync("Drone1")


if __name__ == "__main__":
    lidar_scan_and_map()