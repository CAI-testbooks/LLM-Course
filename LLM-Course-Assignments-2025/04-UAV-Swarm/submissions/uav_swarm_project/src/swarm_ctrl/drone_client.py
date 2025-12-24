import airsim
import time
import logging

import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SwarmManager:
    def __init__(self, vehicle_names=["Drone1", "Drone2", "Drone3"]):
        """
        初始化集群管理器
        :param vehicle_names: 必须与 settings.json 中的名称完全一致
        """
        self.client = airsim.MultirotorClient()

        try:
            self.client.confirmConnection()
        except Exception as e:
            logging.error(f"连接 AirSim 失败: {e}. 请确保虚幻引擎环境已运行并在 Play 模式。")
            exit(1)

        self.vehicle_names = vehicle_names
        self.setup_drones()

    def setup_drones(self):
        """初始化每架无人机的控制权"""
        for name in self.vehicle_names:
            self.client.enableApiControl(True, vehicle_name=name)
            self.client.armDisarm(True, vehicle_name=name)
            logging.info(f"[{name}] API 已解锁，准备就绪。")

    def takeoff_all(self):
        """所有无人机同步起飞"""
        logging.info("集群起飞中...")
        futures = [self.client.takeoffAsync(vehicle_name=name) for name in self.vehicle_names]
        for f in futures:
            f.join()
        logging.info("所有无人机已到达起飞高度。")

    def move_to_positions_async(self, positions, velocity=5):
        """
        向多架无人机同时下达移动指令（不阻塞）
        :param positions: 字典 {'Drone1': [x, y, z], ...}
        :return: futures 列表
        """
        futures = []
        for name, pos in positions.items():
            if name in self.vehicle_names:
                f = self.client.moveToPositionAsync(
                    pos[0], pos[1], pos[2],
                    velocity,
                    vehicle_name=name
                )
                futures.append(f)
        return futures

    def move_on_path_all(self, drone_paths, velocity=5):
        """
        让多架无人机各自沿着自己的航点列表飞行（搜索场景核心功能）
        :param drone_paths: 字典 {'Drone1': [[x1,y1,z1], [x2,y2,z2]], ...}
        """
        futures = []
        for name, path in drone_paths.items():
            # 将路径列表转换为 AirSim 所需的 Vector3r 列表
            point_list = [airsim.Vector3r(p[0], p[1], p[2]) for p in path]
            f = self.client.moveOnPathAsync(
                point_list,
                velocity,
                vehicle_name=name,
                drivetrain=airsim.DrivetrainType.ForwardOnly,
                yaw_mode=airsim.YawMode(False, 0)
            )
            futures.append(f)
        return futures

    def get_all_states(self):
        """获取所有无人机的当前位置和速度"""
        states = {}
        for name in self.vehicle_names:
            multirotor_state = self.client.getMultirotorState(vehicle_name=name)
            pos = multirotor_state.kinematics_estimated.position
            states[name] = {
                "x": pos.x_val,
                "y": pos.y_val,
                "z": pos.z_val
            }
        return states

    def land_all(self):
        """一键降落并释放控制权"""
        logging.info("执行集群降落...")
        futures = [self.client.landAsync(vehicle_name=name) for name in self.vehicle_names]
        for f in futures:
            f.join()

        for name in self.vehicle_names:
            self.client.enableApiControl(False, vehicle_name=name)
        logging.info("所有无人机已降落并释放控制权。")

    def draw_llm_zones_in_airsim(self, assignment_results):
        """
        根据 LLM 的分配结果，在 AirSim 场景地面画出区域边界
        """
        # 定义 RGBA 颜色（对应你图中：粉/红, 绿, 蓝）
        colors = [
            [1.0, 0.5, 0.5, 1.0],  # Drone1: 粉红
            [0.5, 1.0, 0.5, 1.0],  # Drone2: 浅绿
            [0.5, 0.5, 1.0, 1.0]  # Drone3: 浅蓝
        ]

        for i, task in enumerate(assignment_results['tasks']):
            b = task['bounds']
            color = colors[i % len(colors)]

            # 定义矩形的四个地面顶点 (Z=0)
            p1 = airsim.Vector3r(b['x_min'], b['y_min'], 0)
            p2 = airsim.Vector3r(b['x_max'], b['y_min'], 0)
            p3 = airsim.Vector3r(b['x_max'], b['y_max'], 0)
            p4 = airsim.Vector3r(b['x_min'], b['y_max'], 0)

            # 连线绘图 (Duration 设为 300秒)
            self.client.simDrawLine(p1, p2, color, thickness=10, duration=300)
            self.client.simDrawLine(p2, p3, color, thickness=10, duration=300)
            self.client.simDrawLine(p3, p4, color, thickness=10, duration=300)
            self.client.simDrawLine(p4, p1, color, thickness=10, duration=300)

            # 在区域中心打印文字标签
            center_pos = airsim.Vector3r((b['x_min'] + b['x_max']) / 2, (b['y_min'] + b['y_max']) / 2, -2)
            self.client.simPrintText(f"Zone: {task['drone_id']}", center_pos.to_Vector3r(), size=10)

    def get_lidar_safe_direction(self, vehicle_name):
        """
        读取雷达数据，判断前方是否有障碍物
        返回一个修正向量：如果安全则返回 (0,0,0)，如果有障碍物则返回避障偏移量
        """
        lidar_data = self.client.getLidarData(lidar_name="Lidar1", vehicle_name=vehicle_name)

        if len(lidar_data.point_cloud) < 3:
            return airsim.Vector3r(0, 0, 0)

        # 将一维列表转为 [x, y, z] 点云
        points = np.array(lidar_data.point_cloud).reshape(-1, 3)

        # 筛选出无人机正前方（例如 X > 0，距离 < 5米）的障碍点
        # AirSim 坐标系中，X 是前方
        danger_zone = points[(points[:, 0] > 0) & (np.linalg.norm(points, axis=1) < 5.0)]

        if len(danger_zone) > 0:
            # 发现障碍物，计算斥力方向（向左或向右偏转）
            avg_y = np.mean(danger_zone[:, 1])
            avoid_dir = -1.0 if avg_y > 0 else 1.0  # 如果障碍物偏右，我们就向左闪
            return airsim.Vector3r(0, avoid_dir * 3.0, 0)  # 返回一个横向偏移量

        return airsim.Vector3r(0, 0, 0)

    def avoid_other_drones(self, current_vehicle, all_states, min_dist=3.0):
        """
        简单的相互避障：如果离其他飞机太近，产生一个反向推力
        """
        my_pos = all_states[current_vehicle]
        avoid_vector = airsim.Vector3r(0, 0, 0)

        for other_name, other_pos in all_states.items():
            if other_name == current_vehicle:
                continue

            dist = np.sqrt((my_pos['x'] - other_pos['x']) ** 2 + (my_pos['y'] - other_pos['y']) ** 2)
            if dist < min_dist:
                # 计算逃离方向
                avoid_vector.x_val += (my_pos['x'] - other_pos['x']) * 2.0
                avoid_vector.y_val += (my_pos['y'] - other_pos['y']) * 2.0

        return avoid_vector
# 单元测试逻辑
if __name__ == "__main__":
    manager = SwarmManager()
    manager.takeoff_all()

    # 测试获取状态
    print("当前坐标:", manager.get_all_states())

    time.sleep(2)
    manager.land_all()