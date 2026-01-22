import numpy as np
from uav_controller import SimpleMultiUAVController


class UAVAvoidObstacle:
    """避障类：优化参数+统一高度，解决密集障碍物碰撞问题"""
    def __init__(self, obstacle_radius=5, k_att=0.8, k_rep=5):  # 核心参数调整
        self.obstacle_radius = obstacle_radius  # 增大避障半径到5米
        self.k_att = k_att                      # 增大引力（优先向目标移动）
        self.k_rep = k_rep                  # 减小斥力（避免过度偏离路径）



    def calculate_force(self, current_pos, target_pos, obstacles):
        """计算势场合力，得到避障后的移动方向"""
        current_pos = np.array(current_pos)
        target_pos = np.array(target_pos)

        # 统一高度：将目标点Z轴改为当前无人机高度（避免上下飞撞地面）
        target_pos[2] = current_pos[2]

        # 1. 引力：指向巡检点
        att_vector = target_pos - current_pos
        if np.linalg.norm(att_vector) > 1e-6:  # 避免除以0
            att_force = self.k_att * (att_vector / np.linalg.norm(att_vector))
        else:
            att_force = np.zeros(3)  # 已到达目标点

        # 2. 斥力：远离障碍物
        rep_force = np.zeros(3)
        for obs in obstacles:
            obs = np.array(obs)
            obs_vector = current_pos - obs
            obs_dist = np.linalg.norm(obs_vector)

            # 仅对近距离障碍物产生斥力
            if 1e-6 < obs_dist < self.obstacle_radius:
                rep_force += self.k_rep * (1 / obs_dist - 1 / self.obstacle_radius) * (obs_vector / obs_dist)

        # 3. 总合力（归一化）
        total_force = att_force + rep_force
        if np.linalg.norm(total_force) > 1e-6:
            total_force = total_force / np.linalg.norm(total_force)
        return total_force

    def get_surrounding_obstacles(self, current_uav_name, controller: SimpleMultiUAVController, static_obstacles):
        """
        获取当前无人机周围的所有障碍物（静态障碍物 + 其他无人机）
        :param current_uav_name: 当前无人机名称（如UAV_0）
        :param controller: 无人机控制器实例（SimpleMultiUAVController）
        :param static_obstacles: 静态障碍物坐标列表 [[x,y,z], ...]
        :return: 所有障碍物坐标列表 [[x,y,z], ...]
        """
        # 1. 获取当前无人机坐标
        current_pos = controller.get_uav_position(current_uav_name)
        # 2. 初始化障碍物列表（先加入静态障碍物）
        all_obstacles = static_obstacles.copy()

        # 3. 加入其他无人机作为动态障碍物（避免碰撞友机）
        for uav in controller.uav_list:
            if uav != current_uav_name:  # 排除自身
                other_uav_pos = controller.get_uav_position(uav)
                # 仅添加10米内的友机作为障碍物
                if np.linalg.norm(current_pos - other_uav_pos) < 10:
                    all_obstacles.append(other_uav_pos.tolist())

        return all_obstacles

    def move_with_avoidance(self, controller: SimpleMultiUAVController, uav_name, target_pos, static_obstacles, speed=1,
                            step_size=0.5):
        """避障移动到目标点：分步调整方向，直到到达"""
        current_pos = controller.get_uav_position(uav_name)
        target_pos = np.array(target_pos)
        collision_flag = False

        # 距离目标点小于1米则视为到达
        while np.linalg.norm(current_pos - target_pos) > 1:
            # 检测碰撞
            if controller.check_collision(uav_name):
                collision_flag = True
                break

            # 获取周围障碍物
            obstacles = self.get_surrounding_obstacles(uav_name, controller, static_obstacles)

            # 计算避障方向
            force = self.calculate_force(current_pos, target_pos, obstacles)

            # 计算下一步位置（保持高度不变）
            next_pos = current_pos + force * step_size
            next_pos[2] = current_pos[2]  # 固定巡检高度

            # 移动到下一步
            controller.move_to_target(uav_name, next_pos.tolist(), speed, block=True)

            # 更新当前位置
            current_pos = controller.get_uav_position(uav_name)

        if collision_flag:
            print(f"❌ {uav_name} 避障失败（碰撞），未到达目标点")
        else:
            print(f"✅ {uav_name} 避障完成，到达巡检点 {target_pos.tolist()}")

        return not collision_flag  # 返回是否成功到达