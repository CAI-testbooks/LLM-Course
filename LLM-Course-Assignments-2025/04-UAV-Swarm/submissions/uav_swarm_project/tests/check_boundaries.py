import airsim
import time


def step1_check_boundary():
    print("1. 连接 AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()

    vehicle_name = "Drone1"  # 请确保和 settings.json 里的一致

    # 2. 获取控制权 (关键步骤)
    print(f"2. 正在获取 {vehicle_name} 控制权...")
    client.enableApiControl(True, vehicle_name)
    client.armDisarm(True, vehicle_name)
    time.sleep(1)  # 等一秒，确保控制权生效

    # 3. 起飞
    print("3. 起飞中 (Takeoff)...")
    client.takeoffAsync(vehicle_name=vehicle_name).join()

    # 爬升到 5 米高度 (AirSim Z轴向下为正，所以是 -5)
    print("   调整高度到 5米...")
    client.moveToPositionAsync(0, 0, -5, 3, vehicle_name=vehicle_name).join()

    # 4. 飞向边缘测试
    # 如果缩放是 8，地图宽就是 80米，半径是 40米。
    # 我们让它飞到 X=42米，理论上应该刚好飞出地板一点点。
    target_x = 42
    print(f"4. 开始飞向边缘测试点: X={target_x}, Y=0...")

    # 使用 moveToPositionAsync 但不加 .join()，而是手动循环监控，防止卡死
    client.moveToPositionAsync(target_x, 0, -5, 4, vehicle_name=vehicle_name)

    while True:
        # 获取当前位置
        pos = client.getMultirotorState(vehicle_name=vehicle_name).kinematics_estimated.position
        dist = pos.x_val
        print(f"   -> 当前 X 位置: {dist:.1f} 米")

        # 判断是否到达
        if dist >= target_x - 1:
            print(">>> 已到达目标点附近！")
            break

        # # 简单的碰撞检测（防止撞墙卡死）
        # collision = client.simGetCollisionInfo(vehicle_name=vehicle_name)
        # if collision.has_collided:
        #     print(f"!!! 发生碰撞，停止测试。碰撞物体: {collision.object_name}")
        #     break
        #
        # time.sleep(1)

    print("5. 测试完成，悬停 5 秒供观察...")
    print("请现在看 AirSim 窗口：无人机是在地板上方，还是在地板外面的虚空？")
    client.hoverAsync(vehicle_name=vehicle_name)
    time.sleep(5)

    print("6. 降落...")
    client.landAsync(vehicle_name=vehicle_name).join()
    client.armDisarm(False, vehicle_name=vehicle_name)
    client.enableApiControl(False, vehicle_name=vehicle_name)


if __name__ == "__main__":
    step1_check_boundary()