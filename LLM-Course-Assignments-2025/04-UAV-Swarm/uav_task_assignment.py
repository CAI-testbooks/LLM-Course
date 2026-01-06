import numpy as np


def assign_tasks(uav_positions, task_points, covered_points, alpha=0.9):
    """
    动态优先级任务分配：
    - 未覆盖任务权重更高（alpha=0.9）
    - 已覆盖任务权重降低，避免重复巡检
    - 新增负载均衡机制，避免无人机任务过载
    """
    num_uavs = len(uav_positions)
    num_tasks = len(task_points)
    assigned_tasks = [None] * num_uavs
    remaining_tasks = list(range(num_tasks))

    # 为任务分配优先级（未覆盖任务优先级×alpha权重）
    task_priorities = np.ones(num_tasks)
    for t in covered_points:
        if t < num_tasks:  # 防止索引越界
            task_priorities[t] = 1 - alpha  # 已覆盖任务优先级降低

    # 基于位置距离和任务优先级分配，增加负载均衡
    task_load = [0] * num_uavs  # 跟踪每个无人机的任务负载

    for _ in range(min(num_uavs, num_tasks)):  # 每个无人机至少尝试分配一个任务
        best_assignments = []
        for uav_id in range(num_uavs):
            if not remaining_tasks:
                break
            # 计算加权距离（距离×(1/优先级)，优先级越高，距离权重越低）
            dists = []
            for t in remaining_tasks:
                raw_dist = np.linalg.norm(uav_positions[uav_id] - task_points[t])
                # 结合距离、优先级和当前负载计算综合成本
                weighted_dist = raw_dist * (1 / task_priorities[t]) * (1 + 0.1 * task_load[uav_id])
                dists.append(weighted_dist)
            best_task_idx = np.argmin(dists)
            best_task = remaining_tasks[best_task_idx]
            best_cost = dists[best_task_idx]
            best_assignments.append((best_cost, uav_id, best_task, best_task_idx))

        if not best_assignments:
            break

        # 选择全局最优分配
        best_assignments.sort()
        best_cost, uav_id, best_task, best_task_idx = best_assignments[0]

        if assigned_tasks[uav_id] is None:
            assigned_tasks[uav_id] = best_task
            task_load[uav_id] += 1
        else:
            # 如果无人机已有任务，比较新旧任务优先级
            current_task = assigned_tasks[uav_id]
            if task_priorities[best_task] > task_priorities[current_task]:
                # 将旧任务放回待分配列表
                remaining_tasks.append(current_task)
                assigned_tasks[uav_id] = best_task

        del remaining_tasks[best_task_idx]

    # 剩余任务强制分配给最近的无人机（确保全覆盖）
    if remaining_tasks:
        for t in remaining_tasks:
            task_pos = task_points[t]
            uav_dists = [np.linalg.norm(pos - task_pos) for pos in uav_positions]
            nearest_uav = np.argmin(uav_dists)
            # 检查该无人机是否已有任务，且已有任务优先级更低
            current_task = assigned_tasks[nearest_uav]
            if current_task is None or (current_task in covered_points and t not in covered_points):
                assigned_tasks[nearest_uav] = t  # 覆盖旧任务，优先处理新任务
            else:
                # 寻找次优无人机
                sorted_uavs = np.argsort(uav_dists)
                for uav in sorted_uavs[1:]:
                    current_task_uav = assigned_tasks[uav]
                    if current_task_uav is None or (current_task_uav in covered_points and t not in covered_points):
                        assigned_tasks[uav] = t
                        break

    return assigned_tasks


# 测试任务分配（可选执行）
if __name__ == "__main__":
    uav_pos = np.array([[0, 0, -20], [5, 0, -20], [10, 0, -20]])
    from uav_data_generator import generate_task_points

    task_pts = generate_task_points(10)
    assigned = assign_tasks(uav_pos, task_pts, covered_points={0, 1})  # 测试已覆盖任务的分配
    print("动态任务分配结果：", assigned)