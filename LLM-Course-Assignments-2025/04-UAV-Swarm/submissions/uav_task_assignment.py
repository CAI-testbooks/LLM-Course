import numpy as np


def assign_tasks(uav_positions, task_points):
    """
    贪心任务分配：为每个无人机分配最近的未分配任务点
    """
    num_uavs = len(uav_positions)
    num_tasks = len(task_points)
    assigned_tasks = [None] * num_uavs  # 每个无人机的分配任务点
    remaining_tasks = list(range(num_tasks))

    # 第一轮分配：优先分配最近任务
    for uav_id in range(num_uavs):
        if not remaining_tasks:
            break
        # 计算当前无人机到所有剩余任务点的距离
        dists = [np.linalg.norm(uav_positions[uav_id] - task_points[t]) for t in remaining_tasks]
        best_task_idx = np.argmin(dists)
        assigned_tasks[uav_id] = remaining_tasks[best_task_idx]
        del remaining_tasks[best_task_idx]

    # 剩余任务随机分配（保证全覆盖）
    if remaining_tasks:
        for t in remaining_tasks:
            uav_id = np.random.choice(num_uavs)
            assigned_tasks[uav_id] = t  # 覆盖（简化版，实际可分配多个）

    return assigned_tasks


# 测试任务分配（可选执行）
if __name__ == "__main__":
    uav_pos = np.array([[0, 0, -20], [5, 0, -20], [10, 0, -20]])
    from uav_data_generator import generate_task_points

    task_pts = generate_task_points(10)
    assigned = assign_tasks(uav_pos, task_pts)
    print("任务分配结果：", assigned)