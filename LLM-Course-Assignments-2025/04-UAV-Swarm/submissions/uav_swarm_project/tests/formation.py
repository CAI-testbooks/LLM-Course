import numpy as np

class TriangleFormation:
    def __init__(self, spacing=5.0):
        self.spacing = spacing
        self.shape_type = "TRIANGLE"
        # 定义三角形相对位置 (相对于中心点)
        # Drone1(领航), Drone2(左后), Drone3(右后)
        self.offsets = {
            "Drone1": np.array([spacing * 0.866, 0, 0]),     # 前顶点
            "Drone2": np.array([-spacing * 0.5, -spacing, 0]), # 左后
            "Drone3": np.array([-spacing * 0.5, spacing, 0])   # 右后
        }

    def update_spacing(self, factor):
        """LLM 微调接口：调整间距"""
        self.spacing *= factor
        # 重新计算 offset ... (略，按比例缩放)
        print(f"【系统】编队间距已调整为: {self.spacing:.1f} 米")

    def get_target_position(self, swarm_center_pos, drone_name):
        """计算某架飞机在队形里的'理论完美位置'"""
        return swarm_center_pos + self.offsets.get(drone_name, np.zeros(3))