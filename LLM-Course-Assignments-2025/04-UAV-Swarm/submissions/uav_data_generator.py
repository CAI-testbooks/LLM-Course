import numpy as np


def generate_task_points(num_points, x_range=(-50, 50), y_range=(-50, 50), z=-20):
    """
    生成指定数量的任务点坐标
    :param num_points: 任务点数量
    :param x_range: x坐标范围
    :param y_range: y坐标范围
    :param z: z坐标（固定高度）
    :return: 任务点列表，每个元素为[x, y, z]
    """
    task_points = []
    for _ in range(num_points):
        x = np.random.uniform(x_range[0], x_range[1])
        y = np.random.uniform(y_range[0], y_range[1])
        task_points.append(np.array([x, y, z]))
    return task_points


# 测试任务点生成（可选执行）
if __name__ == "__main__":
    points = generate_task_points(10)
    print("生成的任务点：")
    for i, pt in enumerate(points):
        print(f"任务点{i}: {pt}")