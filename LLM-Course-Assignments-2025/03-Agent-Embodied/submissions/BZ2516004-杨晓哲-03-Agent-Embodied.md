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

## 2025-12-29
- Gazebo：无人机模型从简化方块替换为 Iris，修复 spawn/显示问题，demo 可运行。
- Agent：新增/完善 rule-agent 节点（llm_agent_node）输出 plan；新增/完善 plan 执行器（executor_plan）。
- 重播：支持 reset 后重新播放（payload reset / uav reset target 等）。
- 产出：Gazebo 飞行投送流程可复现，日志/轨迹可记录。

