import numpy as np
import airsim


def get_uav_state(client, uav_name, task_points, assigned_tasks):
    """
    改进的状态获取：考虑多个任务点信息
    """
    # 1. 基础状态
    state = client.getMultirotorState(vehicle_name=uav_name)
    pos = np.array([state.kinematics_estimated.position.x_val,
                    state.kinematics_estimated.position.y_val,
                    state.kinematics_estimated.position.z_val])
    vel = np.array([state.kinematics_estimated.linear_velocity.x_val,
                    state.kinematics_estimated.linear_velocity.y_val,
                    state.kinematics_estimated.linear_velocity.z_val])

    # 2. 最近任务点距离（考虑所有分配的任务）
    if assigned_tasks and assigned_tasks[0] is not None:
        task_dists = [np.linalg.norm(pos - task_points[t]) for t in assigned_tasks if t is not None]
        nearest_task_dist = min(task_dists) if task_dists else 100.0
    else:
        nearest_task_dist = 100.0

    # 3. 增强的碰撞风险检测
    collision_risk = 0.0
    # 检测与其他无人机的距离（更灵敏）
    for other_uav in ["UAV0", "UAV1", "UAV2"]:
        if other_uav == uav_name:
            continue
        other_state = client.getMultirotorState(vehicle_name=other_uav)
        other_pos = np.array([other_state.kinematics_estimated.position.x_val,
                              other_state.kinematics_estimated.position.y_val,
                              other_state.kinematics_estimated.position.z_val])
        dist = np.linalg.norm(pos - other_pos)

        # 距离<8米开始有风险，<3米风险极高
        if dist < 8:
            collision_risk = max(collision_risk, min(1.0, 1 - (dist / 8)))

    # 整合状态
    uav_state = np.concatenate([pos, vel, [nearest_task_dist, collision_risk]])
    return uav_state


def avoid_collision(client, uav_name, action, collision_risk, uav_states):
    """
    增强版避障：考虑其他无人机的运动方向
    """
    # 1. 基础避障
    if collision_risk > 0.7:  # 风险阈值降低，更早响应
        # 根据风险等级动态调整避障强度
        action = -action * (0.5 + (1 - collision_risk) * 0.5)

    # 2. 方向避障：远离其他无人机的运动方向
    current_pos = uav_states[:3]
    for other_uav in ["UAV0", "UAV1", "UAV2"]:
        if other_uav == uav_name:
            continue
        other_state = client.getMultirotorState(vehicle_name=other_uav)
        other_pos = np.array([other_state.kinematics_estimated.position.x_val,
                              other_state.kinematics_estimated.position.y_val,
                              other_state.kinematics_estimated.position.z_val])
        other_vel = np.array([other_state.kinematics_estimated.linear_velocity.x_val,
                              other_state.kinematics_estimated.linear_velocity.y_val,
                              other_state.kinematics_estimated.linear_velocity.z_val])

        dist = np.linalg.norm(current_pos - other_pos)
        if dist < 6:  # 近距离避障
            # 计算远离方向
            avoid_dir = current_pos - other_pos
            avoid_dir = avoid_dir / (np.linalg.norm(avoid_dir) + 1e-6)
            # 结合对方速度预判碰撞
            action = action * 0.3 + avoid_dir * 0.7

    # 限制动作范围（降低最大速度）
    action = np.clip(action, -1.5, 1.5)  # 从-2到2调整为-1.5到1.5
    return action


def distributed_path_planning(client, uav_name, maddpg, uav_state, assigned_tasks, task_points):
    """
    改进的路径规划：多任务点动态切换
    """
    # 1. 确定当前目标任务点（完成当前任务后切换到下一个）
    current_pos = uav_state[:3]
    target_task = None
    if assigned_tasks:
        for task in assigned_tasks:
            task_pos = task_points[task]
            if np.linalg.norm(current_pos - task_pos) > 5:  # 未完成的任务
                target_task = task
                break
        # 如果所有任务都完成，取最近的未覆盖任务
        if target_task is None:
            target_task = assigned_tasks[0] if assigned_tasks else None

    # 2. MARL预测动作
    action = maddpg.get_action(int(uav_name[-1]), uav_state)

    # 3. 避障调整
    collision_risk = uav_state[7]
    action = avoid_collision(client, uav_name, action, collision_risk, uav_state)

    # 4. 向目标任务点修正（更强的导向性）
    if target_task is not None:
        target_pos = task_points[target_task]
        target_dir = (target_pos - current_pos) / (np.linalg.norm(target_pos - current_pos) + 1e-6)
        action = action * 0.05 + target_dir * 0.95  # 增强目标导向

    return action, target_task