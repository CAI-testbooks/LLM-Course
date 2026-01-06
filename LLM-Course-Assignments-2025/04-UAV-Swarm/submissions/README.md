# 04-UAV-Swarm
本项目为无人机集群协同巡检系统，基于多智能体强化学习（MADDPG）完成路径规划、避障、任务分配等核心功能，适配 AirSim 仿真环境，支持视频录制与指标评估。
一、环境配置
1. 依赖安装
bash
运行
pip install airsim torch==2.0.1 numpy==1.24.3 opencv-python==4.8.0.74 imageio
2. AirSim 配置
下载 AirSim 预编译包并运行AirSimNH.exe；
替换Documents/AirSim/settings.json：
json
{
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "Vehicles": {
    "UAV0": { "VehicleType": "SimpleFlight", "X": 0, "Y": 0, "Z": -20 },
    "UAV1": { "VehicleType": "SimpleFlight", "X": 5, "Y": 0, "Z": -20 },
    "UAV2": { "VehicleType": "SimpleFlight", "X": 10, "Y": 0, "Z": -20 }
  },
  "CollisionDetection": { "Enabled": true }
}
二、快速开始
启动AirSimNH.exe，等待场景加载；
（可选）后台录制视频：python uav_video_recorder.py；
运行巡检 Demo：python uav_swarm_patrol_demo.py；
（可选）视频转 GIF：脚本自动完成，或调用video_to_gif_hd()函数。
三、核心功能
MARL 模型：简化版 MADDPG，支持奖励函数迭代优化
任务分配：贪心算法分配巡检任务点，保证全覆盖
路径规划 + 避障：分布式规划路径，碰撞风险高时反向减速
通信模拟：模拟 10% 丢包 + 0.1 秒通信延迟
部署与评估：AirSim 集群巡检，自动计算覆盖效率、碰撞率、任务完成时间
视频录制：适配新旧版 AirSim，录制视频 / GIF
四、文件结构
plaintext
├── uav_data_generator.py      # 轨迹数据生成
├── uav_maddpg_model.py        # MADDPG模型实现
├── uav_task_assignment.py     # 任务分配逻辑
├── uav_path_planning.py       # 路径规划+避障
├── uav_communication_sim.py   # 通信模拟
├── uav_swarm_patrol_demo.py   # 核心巡检Demo
├── uav_video_recorder.py      # 视频录制/GIF转换
└── README.md                  # 项目说明