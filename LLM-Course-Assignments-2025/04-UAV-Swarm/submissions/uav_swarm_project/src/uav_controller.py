import cosysairsim as airsim
import numpy as np


class SimpleMultiUAVController:
    """无人机基础控制类：负责起飞、获取坐标、移动、碰撞检测等基础操作"""

    def __init__(self, uav_list):
        # 连接AirSim仿真环境
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.uav_list = uav_list
        print(f"已连接AirSim，无人机列表：{self.uav_list}")

    def takeoff_all(self, height=5):
        """所有无人机起飞并悬停到指定高度（AirSim中Z轴为负，height为正数）"""
        for uav in self.uav_list:
            # 启用API控制
            self.client.enableApiControl(True, uav)
            # 解锁电机
            self.client.armDisarm(True, uav)
            # 起飞
            self.client.takeoffAsync(vehicle_name=uav).join()
            # 悬停到指定高度
            self.client.moveToZAsync(-height, 1, vehicle_name=uav).join()
            print(f"✅ {uav} 已起飞并悬停至 {height} 米高度")

    def get_uav_position(self, uav_name):
        """获取单架无人机的当前坐标（返回numpy数组：[x,y,z]）"""
        state = self.client.getMultirotorState(vehicle_name=uav_name)
        pos = state.kinematics_estimated.position
        return np.array([pos.x_val, pos.y_val, pos.z_val])

    def move_to_target(self, uav_name, target_pos, speed=1, block=True):
        """控制无人机移动到目标坐标"""
        move_task = self.client.moveToPositionAsync(
            target_pos[0], target_pos[1], target_pos[2], speed, vehicle_name=uav_name
        )
        if block:
            move_task.join()
            print(f"📍 {uav_name} 已到达目标点: {target_pos}")

    def check_collision(self, uav_name, min_move_dist=0.5):
        """
        检测无人机是否碰撞（优化：排除初始位置误报，仅移动超过阈值才检测）
        :param uav_name: 无人机名称
        :param min_move_dist: 最小移动距离（米），低于此值不检测碰撞
        :return: 是否真的碰撞（bool）
        """
        # 记录无人机初始位置（首次调用时保存）
        if not hasattr(self, 'init_pos_dict'):
            self.init_pos_dict = {}
        if uav_name not in self.init_pos_dict:
            self.init_pos_dict[uav_name] = self.get_uav_position(uav_name)

        # 计算当前位置与初始位置的距离
        current_pos = self.get_uav_position(uav_name)
        move_dist = np.linalg.norm(current_pos - self.init_pos_dict[uav_name])

        # 仅当移动超过阈值时，才检测碰撞（避免初始位置误报）
        if move_dist < min_move_dist:
            return False

        # 原始碰撞检测逻辑
        collision_info = self.client.simGetCollisionInfo(vehicle_name=uav_name)
        if collision_info.has_collided:
            print(f"⚠️ {uav_name} 发生碰撞（移动距离：{move_dist:.2f}米）！")
            return True
        return False

    def land_all(self):
        """所有无人机降落"""
        for uav in self.uav_list:
            self.client.landAsync(vehicle_name=uav).join()
            # 关闭API控制
            self.client.enableApiControl(False, uav)
            print(f"🛬 {uav} 已降落并关闭API控制")

    def get_dynamic_obstacles(self, obstacle_keywords=None):
        """
        动态获取场景中的障碍物坐标（优化：仅取15米内的关键障碍物，减少数量）
        """
        # 1. 获取场景中所有对象名称（省略打印部分，保持原有逻辑）
        all_objects = self.client.simListSceneObjects()
        if obstacle_keywords is None:
            obstacle_keywords = ["Cone", "Cylinder", "TemplateCube"]

        # 2. 先筛选出所有候选障碍物（排除无关对象）
        candidate_obstacles = []
        exclude_keywords = ["UAV", "Ground", "Sky", "Light", "Weather", "Menu", "Camera", "Game", "World", "Physics",
                            "PostProcess"]
        for obj_name in all_objects:
            if any(ex_key in obj_name for ex_key in exclude_keywords):
                continue
            if any(key in obj_name for key in obstacle_keywords):
                candidate_obstacles.append(obj_name)

        # 3. 仅保留无人机初始位置15米内的障碍物（核心优化）
        obstacle_positions = []
        # 获取任意一架无人机的初始位置（所有无人机初始位置相同）
        if self.uav_list:
            init_uav_pos = self.get_uav_position(self.uav_list[0])
            for obj_name in candidate_obstacles:
                try:
                    pose = self.client.simGetObjectPose(obj_name)
                    pos = pose.position
                    obj_pos = np.array([pos.x_val, pos.y_val, pos.z_val])
                    # 计算与无人机初始位置的水平距离（忽略Z轴）
                    horiz_dist = np.linalg.norm(init_uav_pos[:2] - obj_pos[:2])
                    # 仅保留15米内的障碍物，且数量不超过20个
                    if horiz_dist < 15 and len(obstacle_positions) < 20:
                        obstacle_positions.append([
                            round(pos.x_val, 2),
                            round(pos.y_val, 2),
                            round(pos.z_val, 2)
                        ])
                        print(f"✅ 检测到关键障碍物：{obj_name}，坐标：{obstacle_positions[-1]}（距离：{horiz_dist:.2f}米）")
                except Exception as e:
                    continue
        else:
            print("⚠️ 无无人机列表，无法筛选障碍物距离")

        # 兜底：若仍无障碍物，补充测试用
        if not obstacle_positions:
            obstacle_positions = [[5.0, 5.0, 2.7], [9.0, 7.0, 2.7], [12.0, 4.0, 2.7]]
            print(f"📌 自动补充测试用障碍物：{obstacle_positions}")

        print(f"✅ 筛选后关键障碍物数量：{len(obstacle_positions)}")
        return obstacle_positions