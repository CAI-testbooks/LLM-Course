# 基于LLM的具身智能体（Gazebo UAV Delivery Agent）

- 学号：<BZ2516004>
- 姓名：<杨晓哲>
- 方向：03-Agent-Embodied
- 独立代码仓库链接：<https://github.com/CH-YXZ/ros-embodied-uav-agent>
- Demo 视频链接：<你的录屏链接（后面补）>

## 1. 任务描述
用户一句话指令：无人机飞到投放点 → 降低高度 → 投放包裹 → 上升 → 飞到终点悬停/降落。

## 2. 方法简介（LLM + ReAct工具调用）
- 工具：get_state / takeoff / move_to / descend / drop_payload / hover / land
- 多步规划：自动生成可执行计划（Plan）
- 错误恢复：超时重试、悬停、升高重试、失败降级（返航/降落）
- 记忆模块（可选）：记录失败原因与参数，提升成功率（后期迭代）

## 3. 数据准备（合成轨迹数据）
说明如何在Gazebo中自动生成任务（task_specs）与轨迹（trace.jsonl）。

## 4. 部署与复现
运行步骤见独立代码仓库 README。

## 5. 评估结果
| Split | Success Rate | Avg Steps | Avg Time(s) |
|------|--------------|----------|-------------|
| Test | 待补         | 待补     | 待补        |

## 6. Demo截图/视频
- 视频：<链接>
- 截图：<可插入图片或链接>

# 03-Agent-Embodied 进度日志（BZ2516004 杨晓哲）

## 2025-12-26
### 1) 课程仓库提交与 PR
- 在课程仓库 LLM-Course 创建分支 `hw03-agent-embodied`，按要求提交报告并发起 PR（#23）。
- 修正报告文件命名为课程要求格式，并更新 PR 标题/描述（附独立仓库链接、Demo 链接占位）。

### 2) Gazebo 基础 Demo（可跑通）
- 完成 Gazebo 场景与基础实体生成（无人机 + payload）。
- 初版执行器可驱动无人机完成“起飞→移动→下降→投放→继续飞行→降落”的流程。
- 日志与轨迹输出：生成 `trace_*.json`，用于复现与评估。

## 2025-12-27
### 1) Agent 初版：llm_agent_node.py（规则版）
- 新增 `llm_agent_node.py`：实现“规则版 Agent”，将用户意图（目标点/投送点/高度/等待时间等）转为可执行 plan（JSON/字典结构）。
- Agent 输出 plan 后由执行器逐步调用 ROS 工具链（起飞/移动/下降/悬停/投放/继续飞行/降落），形成“感知/规划/行动”的最小闭环雏形。
- 预留接口：后续可将规则版替换为真实 LLM（ReAct/Tool Calling），Agent 只需改 plan 生成模块，执行器保持不变。

### 2) 可重播流程与调试
- 实践 Gazebo reset / 重新播放流程：支持 reset 后再次执行任务（重复实验）。
- 修正 launch 注释方式与 XML 语法问题（避免 `Invalid roslaunch XML syntax`）。
- 处理模型/节点名不一致导致的控制报错：`GetModelState: model [uav_dummy] does not exist`（统一 spawn 的 model name 与控制端读取名称）。

### 3) 动作节奏优化（更符合任务描述）
- 调整飞行速度与节奏，使动作可观察：加入 wait/hover/指定耗时移动。
- 目标动作节奏（示例）：
  - 起点等待 5s → 起飞到 1m
  - 前飞 5m 用时 5s
  - 降到 0.5m 悬停 3s → 投包
  - 升到 1m，再前飞 5m 用时 5s
  - 最后降落


---

## 2025-12-29
- Gazebo：无人机模型从简化方块替换为 Iris，修复 spawn/显示问题，demo 可运行。
- Agent：新增/完善 rule-agent 节点（llm_agent_node）输出 plan；新增/完善 plan 执行器（executor_plan）。
- 重播：支持 reset 后重新播放（payload reset / uav reset target 等）。
- 产出：Gazebo 飞行投送流程可复现，日志/轨迹可记录。

##2026-01-04
-更新操作步骤：
cd ~/ros-embodied-uav-agent/ros_ws
source devel/setup.bash
roslaunch uav_gazebo_demo agent_demo.launch
#这时Gazebo运行

--新开一个终端：
rostopic pub -1 /agent/command std_msgs/String \
"data: '起点等待5秒，起飞到1米，向前飞5米用时5秒，降到0.5米悬停3秒扔包，升到1米向前飞5米用时5秒，最后降落'"

-重播（reset 后再来一遍）：
rosservice call /demo/replay

-然后再发一次指令：
rostopic pub -1 /agent/command std_msgs/String \
"data: '起点等待5秒，起飞到1米，向前飞5米用时5秒，降到0.5米悬停3秒扔包，升到1米向前飞5米用时5秒，最后降落'"


-查看执行日志/轨迹（trace）：

ls -lh ~/ros-embodied-uav-agent/ros_ws/src/uav_gazebo_demo/results/traces
tail -n 30 ~/ros-embodied-uav-agent/ros_ws/src/uav_gazebo_demo/results/traces/*.jsonl

##2026-01-04
-更新replay_service.py程序，解决bug
重播时无人机的“真实位姿”没有被立刻复位到原点（/gazebo/reset_world 对“运行时 spawn 出来的模型”有时不会回到最初 spawn 位姿。

解决办法：在 /demo/replay 里除了 reset_world，再显式调用 /gazebo/set_model_state 把无人机（和 payload）强制放回初始位姿。

##2026-01-04
-更新llm_agent_node.py程序
根据发的文字内容来改变飞行任务