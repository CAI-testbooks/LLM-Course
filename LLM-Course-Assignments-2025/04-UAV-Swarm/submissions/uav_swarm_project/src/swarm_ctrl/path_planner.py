import numpy as np

class PathPlanner:
    def __init__(self, step_size=4.0, altitude=-5.0):
        self.step_size = step_size  # 扫描间距
        self.altitude = altitude  # 飞行高度

    def generate_lawnmower_path(self, bounds):
        """生成 S 型（割草机）航点"""
        path = []
        x_curr = bounds['x_min']

        # 沿着 X 轴步进
        side_toggle = True
        while x_curr <= bounds['x_max']:
            if side_toggle:
                path.append([x_curr, bounds['y_min'], self.altitude])
                path.append([x_curr, bounds['y_max'], self.altitude])
            else:
                path.append([x_curr, bounds['y_max'], self.altitude])
                path.append([x_curr, bounds['y_min'], self.altitude])

            x_curr += self.step_size
            side_toggle = not side_toggle

        return path


