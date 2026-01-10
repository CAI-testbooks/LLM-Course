import json
import time
from uav_controller import SimpleMultiUAVController
from llm_assigner import LLMUAVTaskAssigner  # 新增Mock类
from obstacle_avoider import UAVAvoidObstacle


def load_config(config_path="../config/settings.json"):
    """加载配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def main():
    # ====================== 1. 加载配置 ======================
    print("📌 加载配置文件...")
    config = load_config()

    # ====================== 2. 初始化模块 ======================
    print("\n📌 初始化无人机控制器...")
    controller = SimpleMultiUAVController(config["uav_list"])

    print("\n📌 初始化硅基流动LLM任务分配器...")
    llm_assigner = LLMUAVTaskAssigner(
        api_key=config["siliconflow_api_key"],  # 硅基流动API Key
        base_url=config["siliconflow_base_url"],  # 硅基流动base_url
        model=config["llm_model"]
    )

    print("\n📌 初始化避障模块...")
    obstacle_avoider = UAVAvoidObstacle(
        obstacle_radius=config["obstacle_radius"]
    )



    # ====================== 4. 动态获取障碍物坐标（核心修改） ======================
    print("\n📌 动态获取场景中的障碍物坐标...")
    # 核心修改：匹配场景中的Cone/Cylinder/TemplateCube（排除无人机/无关对象）
    obstacle_keywords = ["Cone", "Cylinder", "TemplateCube"]
    dynamic_obstacles = controller.get_dynamic_obstacles(obstacle_keywords)
    print(f"✅ 共检测到 {len(dynamic_obstacles)} 个障碍物")

    # ====================== 5. 获取无人机当前坐标 ======================
    print("\n📌 获取无人机当前坐标...")
    uav_pos_dict = {}
    for uav in config["uav_list"]:
        uav_pos_dict[uav] = controller.get_uav_position(uav).tolist()
    print(f"无人机当前坐标：{uav_pos_dict}")

    # ====================== 6. LLM分配巡检点 ======================
    print("\n📌 调用LLM分配巡检点...")
    assign_result = llm_assigner.assign_patrol_points(
        uav_pos_dict=uav_pos_dict,
        patrol_points=config["patrol_points"]
    )
    if not assign_result:
        print("❌ 任务分配失败，程序退出")
        controller.land_all()
        return

    # ====================== 3. 无人机起飞 ======================
    print("\n📌 无人机起飞...")
    controller.takeoff_all(height=config["patrol_height"])
    time.sleep(1)  # 起飞后等待1秒

    # ====================== 7. 无人机避障前往巡检点 ======================
    print("\n📌 无人机开始避障前往巡检点...")
    success_count = 0  # 成功到达的无人机数量
    start_time = time.time()

    for uav in config["uav_list"]:
        if uav not in assign_result:
            print(f"❌ {uav} 无分配的巡检点，跳过")
            continue
        target_point = assign_result[uav]
        print(f"\n🔹 {uav} 前往巡检点：{target_point}")
        # 避障移动：使用动态获取的障碍物
        success = obstacle_avoider.move_with_avoidance(
            controller=controller,
            uav_name=uav,
            target_pos=target_point,
            static_obstacles=dynamic_obstacles,  # 替换为动态障碍物
            speed=config["uav_speed"],
            step_size=config["step_size"]
        )
        if success:
            success_count += 1

    # ====================== 8. 任务完成统计 ======================
    end_time = time.time()
    total_time = round(end_time - start_time, 2)
    completion_rate = round((success_count / len(config["uav_list"])) * 100, 2)

    print("\n" + "=" * 50)
    print("📊 任务完成统计")
    print("=" * 50)
    print(f"总飞行时间：{total_time} 秒")
    print(f"成功到达巡检点的无人机数：{success_count}/{len(config['uav_list'])}")
    print(f"任务完成率：{completion_rate}%")
    print(f"检测到的障碍物数量：{len(dynamic_obstacles)}")
    print("=" * 50)

    # ====================== 9. 无人机返航降落 ======================
    print("\n📌 无人机返航降落...")
    controller.land_all()
    print("\n🎉 测试完成！")


if __name__ == "__main__":
    main()