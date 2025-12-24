import os
import time
import logging
from dotenv import load_dotenv

# 导入我们之前编写的模块
from llm_planner.task_allocator import LLMCommander
from swarm_ctrl.drone_client import SwarmManager
from swarm_ctrl.path_planner import PathPlanner

# 配置日志输出
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [Commander] - %(levelname)s - %(message)s')


def run_coordinated_search():
    # 1. 配置参数
    # 如果你没有使用 .env，可以直接在这里输入 API_KEY
    API_KEY = "sk-pgajauuypprsyzlcctewuurkpctqmnhjrlatbijkbhjjtere"
    DRONE_NAMES = ["Drone1", "Drone2", "Drone3"]
    TOTAL_SEARCH_AREA = {"x_min": 0, "x_max": 60, "y_min": 0, "y_max": 60}

    # 用户指令（可以尝试更改它，看看 LLM 如何反应）
    USER_COMMAND = "现在有三架无人机，请帮我搜索前方 60x60 的正方形区域。请垂直于X轴方向进行均匀切分。"

    # 2. 初始化组件
    commander = LLMCommander(api_key=API_KEY)
    planner = PathPlanner(step_size=5.0, altitude=-5.0)  # 5米间距，5米高度
    manager = SwarmManager(vehicle_names=DRONE_NAMES)

    try:
        # 3. LLM 决策阶段
        logging.info(f"正在向硅基流动 LLM 发送任务指令: {USER_COMMAND}")
        allocation_result = commander.allocate_tasks(USER_COMMAND, DRONE_NAMES, TOTAL_SEARCH_AREA)

        logging.info("LLM 任务分配完成:")
        for task in allocation_result['tasks']:
            logging.info(f" -> {task['drone_id']}: 负责区域 {task['bounds']}")

        # 4. 路径规划阶段
        drone_paths = {}
        for task in allocation_result['tasks']:
            name = task['drone_id']
            bounds = task['bounds']
            # 将 LLM 给出的矩形范围转换为 S 型航点序列
            path = planner.generate_lawnmower_path(bounds)
            drone_paths[name] = path
            logging.info(f"已为 {name} 生成 {len(path)} 个搜索航点")

        # 5. 执行阶段
        logging.info("准备起飞...")
        manager.takeoff_all()
        time.sleep(2)  # 稳定一下

        logging.info("开始下发协同搜索路径...")
        # 核心：多机同时沿路径飞行
        path_futures = manager.move_on_path_all(drone_paths, velocity=6)

        # 6. 监控阶段
        logging.info("搜索中... 正在监控无人机状态")
        while True:
            # 检查是否所有路径都已飞完
            all_done = all([f.is_done() for f in path_futures])
            if all_done:
                logging.info("所有无人机已完成搜索路径。")
                break

            # 实时打印位置（可选）
            states = manager.get_all_states()
            # 这里可以根据需要添加逻辑：比如发现目标点立即停下
            time.sleep(1)

        # 7. 任务结束
        logging.info("任务结束，执行集群降落。")
        manager.land_all()

    except Exception as e:
        logging.error(f"任务执行出错: {e}")
        # 出错时紧急尝试降落
        manager.land_all()


if __name__ == "__main__":
    run_coordinated_search()