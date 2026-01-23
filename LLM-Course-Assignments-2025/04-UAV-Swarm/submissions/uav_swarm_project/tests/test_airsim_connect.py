import airsim

import time


def smoke_test():
    # 1. 连接到 AirSim (Crosys/1.8版本默认端口 41451)
    client = airsim.MultirotorClient()
    try:
        client.confirmConnection()
        print("成功连接到 AirSim 模拟器。")
    except Exception as e:
        print(f"连接失败: {e}")
        return

    # 2. 接管控制权 (API Control)
    vehicle_name = "Drone1"
    client.enableApiControl(True,vehicle_name=vehicle_name)
    client.armDisarm(True,vehicle_name=vehicle_name)
    print("已接管无人机控制权。")
    # 3. 起飞测试 (Takeoff)
    client.takeoffAsync(vehicle_name=vehicle_name).join()
    print("无人机已起飞。")
    # 4. 悬停并移动一段距离 (简单坐标控制)
    hover_duration = 30  # 悬停时间（秒）
    client.moveToPositionAsync(0, 0, -10, 5, vehicle_name=vehicle_name).join()

    # 无人机悬停30秒
    time.sleep(hover_duration)
    print(f"无人机悬停 {hover_duration} 秒。")
    # 5. 降落 (Landing)
    client.landAsync(vehicle_name=vehicle_name).join()
    print("无人机已降落。")
    # 6. 释放控制权
    client.armDisarm(False,vehicle_name=vehicle_name)
    client.enableApiControl(False,vehicle_name=vehicle_name)
    print("已释放无人机控制权。")


if __name__ == "__main__":
    smoke_test()