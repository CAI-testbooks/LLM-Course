import numpy as np


def assign_tasks(uav_positions, task_points):
    """
    改进的任务分配：基于距离和负载均衡的双因素分配
    """
    num_uavs = len(uav_positions)
    num_tasks = len(task_points)
    assigned_tasks = [[] for _ in range(num_uavs)]  # 每个无人机可分配多个任务
    remaining_tasks = list(range(num_tasks))

    # 计算所有无人机到所有任务点的距离矩阵
    dist_matrix = np.zeros((num_uavs, num_tasks))
    for i, uav_pos in enumerate(uav_positions):
        for j, task_pt in enumerate(task_points):
            dist_matrix[i, j] = np.linalg.norm(uav_pos - task_pt)

    # 改进分配策略：最小化总距离 + 均衡任务负载
    while remaining_tasks:
        # 对每个剩余任务，找到距离最近的无人机
        task_distances = []
        for task_idx in remaining_tasks:
            uav_idx = np.argmin(dist_matrix[:, task_idx])
            # 综合考虑距离和当前负载（负载权重0.3）
            cost = dist_matrix[uav_idx, task_idx] * 0.7 + len(assigned_tasks[uav_idx]) * 0.3
            task_distances.append((cost, uav_idx, task_idx))

        # 选择成本最低的分配方案
        task_distances.sort()
        _, best_uav, best_task = task_distances[0]

        assigned_tasks[best_uav].append(best_task)
        remaining_tasks.remove(best_task)

    # 转换为当前框架兼容的格式（主任务+备选任务）
    primary_tasks = []
    for tasks in assigned_tasks:
        primary_tasks.append(tasks[0] if tasks else None)

    return primary_tasks, assigned_tasks  # 返回主任务和完整任务列表


# 测试任务分配
if __name__ == "__main__":
    uav_pos = np.array([[0, 0, -20], [5, 0, -20], [10, 0, -20]])
    from uav_data_generator import generate_task_points

    task_pts = generate_task_points(10)
    primary, all_tasks = assign_tasks(uav_pos, task_pts)
    print("主任务分配：", primary)
    print("所有任务分配：", all_tasks)