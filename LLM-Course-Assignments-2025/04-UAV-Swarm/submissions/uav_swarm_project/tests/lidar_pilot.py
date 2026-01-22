import airsim
import numpy as np


def get_combined_force(client, vehicle_name, target_pos, current_pos, other_drones_pos):
    """
    计算合力：
    F_total = F_goal (飞向队形站位) + F_avoid (雷达避障) + F_swarm (机间防撞)
    """
    # ==========================
    # 1. 引力：飞向编队目标点 (虚拟弹簧力)
    # ==========================
    vec_to_target = target_pos - current_pos
    dist_target = np.linalg.norm(vec_to_target)

    # 修复: 只有距离足够远才产生引力，避免在目标点抖动
    if dist_target > 0.2:
        f_goal = (vec_to_target / dist_target) * 4.0
    else:
        f_goal = np.array([0., 0., 0.])

    # ==========================
    # 2. 斥力：雷达点云避障 (已修复除零错误)
    # ==========================
    lidar_data = client.getLidarData(lidar_name="Lidar1", vehicle_name=vehicle_name)
    f_avoid = np.array([0., 0., 0.])

    # 记录障碍物强度用于 LLM 反思
    obs_intensity = 0.0

    if len(lidar_data.point_cloud) >= 3:
        points = np.array(lidar_data.point_cloud).reshape(-1, 3)

        # --- 关键修复 1: 过滤掉太近的点 (噪音或自身) ---
        # 计算每个点到雷达中心的距离
        dists = np.linalg.norm(points, axis=1)

        # 只保留: 距离 > 0.5米 (过滤自身) 且 < 6.0米 (感知范围) 的点
        valid_mask = (dists > 0.5) & (dists < 6.0)
        danger_points = points[valid_mask]
        danger_dists = dists[valid_mask]

        if len(danger_points) > 0:
            for i, p in enumerate(danger_points):
                d = danger_dists[i]

                # --- 关键修复 2: 数值保护 ---
                # 斥力公式修改：避免分母为0
                # 原始: 1.0 / (d - 0.1)**2  -> 如果 d=0.1 就炸了
                # 修正: 使用 max(d - 0.5, 0.1) 确保分母最小也是 0.1
                safe_dist_factor = max(d - 0.5, 0.1)

                # 斥力方向：-p (从障碍物指向我)
                repulsion = -1.0 * (p / d) * (1.0 / (safe_dist_factor ** 2))
                f_avoid += repulsion

            # 归一化限制最大斥力，防止力过大飞出地图
            force_mag = np.linalg.norm(f_avoid)
            obs_intensity = force_mag  # 记录原始强度给 LLM

            if force_mag > 8.0:
                f_avoid = (f_avoid / force_mag) * 8.0

    # ==========================
    # 3. 机间斥力：防止撞队友 (已修复除零错误)
    # ==========================
    f_swarm = np.array([0., 0., 0.])
    for other_name, other_pos in other_drones_pos.items():
        if other_name == vehicle_name: continue

        diff = current_pos - other_pos
        dist = np.linalg.norm(diff)

        # --- 关键修复 3: 距离过近时的保护 ---
        if dist < 0.1:
            # 如果重叠了，给一个随机方向推开，防止 NaN
            f_swarm += np.random.randn(3) * 10.0
        elif dist < 3.0:  # 3米安全圈
            # 斥力函数
            f_swarm += (diff / dist) * (5.0 / (dist + 0.1))  # +0.1 避免极端情况






    # ==========================
    # 4. 合成与限幅
    # ==========================
    # 权重调整：环境避障优先级最高
    total_force = f_goal * 1.5 + f_avoid * 2.5 + f_swarm * 3.0

    # 速度限幅 (最大 5m/s)
    speed = np.linalg.norm(total_force)
    if speed > 5.0:
        total_force = (total_force / speed) * 5.0

    return total_force, obs_intensity