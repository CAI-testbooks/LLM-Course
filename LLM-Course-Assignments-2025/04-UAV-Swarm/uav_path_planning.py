import numpy as np
import airsim


def get_uav_state(client, uav_name, task_points, assigned_task):  # 确保参数为4个：client, uav_name, task_points, assigned_task
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

    # 2. 最近任务点距离（使用分配的任务点计算）
    if assigned_task is not None and assigned_task < len(task_points):
        nearest_task_dist = np.linalg.norm(pos - task_points[assigned_task])
    else:
        nearest_task_dist = 100.0  # 无任务时设为大值

    # 3. 碰撞风险（检测与其他无人机/环境的距离）
    collision_risk = 0.0
    # 检测与其他无人机的距离
    for other_uav in ["UAV0", "UAV1", "UAV2"]:
        if other_uav == uav_name:
            continue
        try:
            other_pos = client.getMultirotorState(vehicle_name=other_uav).kinematics_estimated.position
            other_pos = np.array([other_pos.x_val, other_pos.y_val, other_pos.z_val])
            dist = np.linalg.norm(pos - other_pos)
            if dist < 5:  # 距离<5米则碰撞风险升高
                collision_risk = max(collision_risk, min(1.0, 1 - dist / 5))
        except:
            continue  # 忽略获取状态失败的无人机

    # 整合状态：位置(x,y,z)+速度(vx,vy,vz)+最近任务点距离+碰撞风险（共8维）
    uav_state = np.concatenate([pos, vel, [nearest_task_dist, collision_risk]])
    return uav_state


def avoid_collision(client, uav_name, action, collision_risk):
    """
    避障调整：高风险时仅横向避障，不影响纵向任务方向
    """
    if collision_risk > 0.8:
        # 仅在x/y方向避障，z方向保持向任务点移动
        action[:2] = -action[:2] * 0.5  # 横向反向减速
    # 适当提高最大速度限制
    action = np.clip(action, -3, 3)
    return action


def distributed_path_planning(client, uav_name, maddpg, uav_state, assigned_task, task_points, covered_points):
    """
    分布式路径规划：增强对未覆盖任务的跟踪精度
    """
    # 1. MARL预测动作
    action = maddpg.get_action(int(uav_name[-1]), uav_state)

    # 2. 避障调整
    collision_risk = uav_state[7]
    action = avoid_collision(client, uav_name, action, collision_risk)

    # 3. 任务点跟踪优化（未覆盖任务权重更高）
    if assigned_task is not None and assigned_task < len(task_points):
        target_pos = task_points[assigned_task]
        current_pos = uav_state[:3]
        dist_to_target = np.linalg.norm(target_pos - current_pos)

        # 未覆盖任务：增强跟踪权重
        if assigned_task not in covered_points:
            target_dir = (target_pos - current_pos) / (dist_to_target + 1e-6)
            # 距离越远，目标方向权重越高（确保快速接近）
            action = action * 0.1 + target_dir * min(0.9, 1.0 - dist_to_target / 50)
        else:
            # 已覆盖任务：降低权重，允许探索新任务
            target_dir = (target_pos - current_pos) / (dist_to_target + 1e-6)
            action = action * 0.5 + target_dir * 0.5

    return action