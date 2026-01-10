import sys
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 路径修复
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from llm_planner.task_allocator import LLMCommander


def run_llm_unit_test():
    # 请填入你的硅基流动 API Key
    API_KEY = "sk-pgajauuypprsyzlcctewuurkpctqmnhjrlatbijkbhjjtere"
    commander = LLMCommander(api_key=API_KEY)

    # 模拟输入场景：100x100 的区域
    total_area = {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 100}
    drones = ["Drone1", "Drone2", "Drone3"]

    # 测试指令：尝试让它不平均分配，看看它聪不聪明
    user_cmd = "Drone1性能最好，给它分一半区域，剩下的给另外两架平均分。"

    print(f"发送指令: {user_cmd}")
    result = commander.allocate_tasks(user_cmd, drones, total_area)

    # 可视化
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-10, 110)
    ax.set_ylim(-10, 110)
    colors = ['#FF9999', '#99FF99', '#9999FF']

    for i, task in enumerate(result['tasks']):
        b = task['bounds']
        rect = patches.Rectangle(
            (b['x_min'], b['y_min']),
            b['x_max'] - b['x_min'],
            b['y_max'] - b['y_min'],
            linewidth=2, edgecolor='black', facecolor=colors[i], alpha=0.6,
            label=f"{task['drone_id']}"
        )
        ax.add_patch(rect)
        # 标注文字
        plt.text((b['x_min'] + b['x_max']) / 2, (b['y_min'] + b['y_max']) / 2,
                 task['drone_id'], ha='center', va='center', fontsize=10, fontweight='bold')

    plt.title(f"LLM Assignment Visualization\nCmd: {user_cmd}")
    plt.xlabel("X Coordinate (meters)")
    plt.ylabel("Y Coordinate (meters)")
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7)
    plt.show()


if __name__ == "__main__":
    run_llm_unit_test()