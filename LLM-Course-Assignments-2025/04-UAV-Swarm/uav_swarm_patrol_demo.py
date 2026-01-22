import numpy as np
import time
import airsim
from uav_data_generator import generate_task_points
from uav_maddpg_model import SimpleMADDPG
from uav_task_assignment import assign_tasks
from uav_path_planning import get_uav_state, distributed_path_planning
from uav_communication_sim import communicate_uav_states


def uav_swarm_patrol():
    # 1. 初始化AirSim客户端
    client = airsim.MultirotorClient()
    client.confirmConnection()
    uav_names = ["UAV0", "UAV1", "UAV2"]

    # 2. 初始化无人机
    for uav in uav_names:
        client.enableApiControl(True, uav)
        client.armDisarm(True, uav)
        client.takeoffAsync(vehicle_name=uav).join()  # 起飞
        client.moveToZAsync(-20, 8, vehicle_name=uav).join()  # 定高20米，更快的上升速度

    # 3. 任务点和初始分配
    task_points = generate_task_points(10)
    uav_positions = [
        np.array([client.getMultirotorState(uav).kinematics_estimated.position.x_val,
                  client.getMultirotorState(uav).kinematics_estimated.position.y_val,
                  client.getMultirotorState(uav).kinematics_estimated.position.z_val])
        for uav in uav_names
    ]
    covered_points = set()
    assigned_tasks = assign_tasks(uav_positions, task_points, covered_points)

    # 4. 初始化MARL模型
    maddpg = SimpleMADDPG(num_uavs=3, state_dim=8, action_dim=3)

    # 5. 巡检参数
    max_steps = 300  # 减少最大步数，提高效率要求
    step = 0.3  # 减小步长，提高响应速度
    collision_count = 0  # 碰撞次数
    start_time = time.time()
    prev_covered = set()  # 跟踪上一步覆盖的点

    # 记录每个任务点的覆盖次数，避免过度覆盖
    task_coverage_count = {i: 0 for i in range(len(task_points))}

    # 6. 集群巡检主循环
    for step_idx in range(max_steps):
        # 6.1 获取所有无人机状态
        uav_states = []
        for uav, task in zip(uav_names, assigned_tasks):
            state = get_uav_state(client, uav, task_points, task)
            uav_states.append(state)

        # 6.1.1 动态更新无人机位置（用于任务分配）
        uav_positions = [state[:3] for state in uav_states]

        # 6.1.2 每2步重分配一次任务（更频繁的动态调整）
        if step_idx % 2 == 0:
            assigned_tasks = assign_tasks(uav_positions, task_points, covered_points)

        # 6.2 通信模拟
        comm_states = communicate_uav_states(uav_states, uav_names)

        # 6.3 分布式路径规划+避障（传入covered_points）
        actions = []
        for uav, state, task in zip(uav_names, uav_states, assigned_tasks):
            action = distributed_path_planning(
                client, uav, maddpg, state, task, task_points, covered_points
            )
            actions.append(action)

        # 6.4 执行动作（控制无人机移动）
        for uav, action in zip(uav_names, actions):
            client.moveByVelocityAsync(
                action[0], action[1], action[2],
                duration=step, vehicle_name=uav
            ).join()

        # 6.5 检测碰撞
        current_collision = False
        for uav in uav_names:
            collision_info = client.simGetCollisionInfo(vehicle_name=uav)
            if collision_info.has_collided:
                collision_count += 1
                current_collision = True
                # 更智能的碰撞后重置：移动到附近安全位置
                current_pos = uav_positions[int(uav[-1])]
                safe_offset = np.random.normal(0, 3, 3)  # 随机偏移
                safe_offset[2] = -20  # 保持高度
                client.moveToPositionAsync(
                    current_pos[0] + safe_offset[0],
                    current_pos[1] + safe_offset[1],
                    safe_offset[2],
                    7, vehicle_name=uav
                ).join()

        # 6.6 检测任务点覆盖（收紧阈值到4米，提高精度）
        current_covered = set()
        for uav, state, task in zip(uav_names, uav_states, assigned_tasks):
            if task is not None and task < len(task_points) and state[6] < 4:
                current_covered.add(task)
                task_coverage_count[task] += 1

        # 合并新覆盖的点，对已多次覆盖的点降低优先级
        newly_covered = current_covered - covered_points
        covered_points.update(current_covered)

        # 6.7 MARL模型训练（传入上一步覆盖情况）
        next_uav_states = [get_uav_state(client, uav, task_points, task) for uav, task in
                           zip(uav_names, assigned_tasks)]
        rewards = maddpg.calculate_reward_optimized(
            uav_states, task_points, covered_points, current_collision, prev_covered
        )
        critic_loss, actor_loss = maddpg.train_step(uav_states, actions, rewards, next_uav_states)

        # 更新上一步覆盖记录
        prev_covered = set(covered_points)

        # 6.8 打印进度
        if step_idx % 10 == 0:
            coverage_rate = len(covered_points) / len(task_points) * 100
            print(
                f"Step {step_idx}: 覆盖率={coverage_rate:.1f}%, 碰撞次数={collision_count}, CriticLoss={critic_loss:.4f}")

        time.sleep(step)

        # 6.9 任务完成（全覆盖则退出）
        if len(covered_points) == len(task_points):
            print("所有任务点覆盖完成！")
            break

    # 7. 结束巡检
    end_time = time.time()
    task_time = end_time - start_time
    for uav in uav_names:
        client.landAsync(vehicle_name=uav).join()
        client.armDisarm(False, uav)
        client.enableApiControl(False, uav)

    # 8. 评估指标计算
    coverage_efficiency = len(covered_points) / len(task_points)
    collision_rate = collision_count / max_steps
    print(f"\n巡检完成！")
    print(f"覆盖效率：{coverage_efficiency:.2f}")
    print(f"碰撞率：{collision_rate:.4f}")
    print(f"任务完成时间：{task_time:.2f}秒")


# 运行集群巡检Demo
if __name__ == "__main__":
    uav_swarm_patrol()