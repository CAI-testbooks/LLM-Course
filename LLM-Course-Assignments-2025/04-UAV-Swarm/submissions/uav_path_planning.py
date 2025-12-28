import numpy as np
import airsim


def get_uav_state(client, uav_name, task_points, assigned_task):
    """
    获取单无人机状态：位置+速度+最近任务点距离+碰撞风险
    """
    # 1. 基础状态
    state = client.getMultirotorState(vehicle_name=uav_name)
    pos = np.array([state.kinematics_estimated.position.x_val,
                    state.kinematics_estimated.position.y_val,
                    state.kinematics_estimated.position.z_val])
    vel = np.array([state.kinematics_estimated.linear_velocity.x_val,
                    state.kinematics_estimated.linear_velocity.y_val,
                    state.kinematics_estimated.linear_velocity.z_val])

    # 2. 最近任务点距离
    if assigned_task is not None:
        nearest_task_dist = np.linalg.norm(pos - task_points[assigned_task])
    else:
        nearest_task_dist = 100.0  # 无任务时设为大值

    # 3. 碰撞风险（检测与其他无人机/环境的距离）
    collision_risk = 0.0
    # 检测与其他无人机的距离
    for other_uav in ["UAV0", "UAV1", "UAV2"]:
        if other_uav == uav_name:
            continue
        other_pos = client.getMultirotorState(vehicle_name=other_uav).kinematics_estimated.position
        other_pos = np.array([other_pos.x_val, other_pos.y_val, other_pos.z_val])
        dist = np.linalg.norm(pos - other_pos)
        if dist < 5:  # 距离<5米则碰撞风险升高
            collision_risk = min(1.0, 1 - dist / 5)

    # 整合状态
    uav_state = np.concatenate([pos, vel, [nearest_task_dist, collision_risk]])
    return uav_state


def avoid_collision(client, uav_name, action, collision_risk):
    """
    避障调整：碰撞风险高时，反向调整动作
    """
    if collision_risk > 0.8:
        action = -action * 0.5  # 反向减速
    # 限制动作范围（避免速度过大）
    action = np.clip(action, -2, 2)
    return action


def distributed_path_planning(client, uav_name, maddpg, uav_state, assigned_task, task_points):
    """
    分布式路径规划：单无人机基于局部状态和MARL动作规划路径
    """
    # 1. MARL预测动作（速度调整）
    action = maddpg.get_action(int(uav_name[-1]), uav_state)

    # 2. 避障调整
    collision_risk = uav_state[7]
    action = avoid_collision(client, uav_name, action, collision_risk)

    # 3. 向任务点修正（分布式规划）
    if assigned_task is not None:
        target_pos = task_points[assigned_task]
        current_pos = uav_state[:3]
        target_dir = (target_pos - current_pos) / (np.linalg.norm(target_pos - current_pos) + 1e-6)
        action = action * 0.1 + target_dir * 0.9  # 结合MARL动作和目标方向

    return action