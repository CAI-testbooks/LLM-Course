import airsim
import time


def test_simultaneous_takeoff():
    # 1. 连接 AirSim
    client = airsim.MultirotorClient()
    client.confirmConnection()

    # 定义无人机名称列表（需与 settings.json 一致）
    drones = ["Drone1", "Drone2", "Drone3"]

    print("--- 准备阶段 ---")
    for name in drones:
        # 获取每架飞机的控制权
        client.enableApiControl(True, vehicle_name=name)
        # 解锁电机
        client.armDisarm(True, vehicle_name=name)
        print(f"{name} 已准备就绪")

    # 2. 执行同时起飞
    print("\n--- 发送同时起飞指令 ---")
    # 创建异步任务列表
    takeoff_tasks = []
    for name in drones:
        # 注意：这里调用的是 takeoffAsync，它不会阻塞程序
        task = client.takeoffAsync(vehicle_name=name)
        takeoff_tasks.append(task)

    # 3. 等待所有任务完成
    print("等待所有无人机到达起飞高度...")
    for task in takeoff_tasks:
        task.join()  # join() 会确保该任务执行完毕

    print("起飞完成！所有无人机已悬停。")

    # 4. 简单悬停保持 5 秒，观察稳定性
    time.sleep(5)

    # 5. 同时降落
    print("\n--- 任务结束：开始降落 ---")
    land_tasks = [client.landAsync(vehicle_name=name) for name in drones]
    for task in land_tasks:
        task.join()

    # 6. 释放控制权
    for name in drones:
        client.enableApiControl(False, vehicle_name=name)
    print("API 控制权已释放。")


if __name__ == "__main__":
    try:
        test_simultaneous_takeoff()
    except Exception as e:
        print(f"发生错误: {e}")