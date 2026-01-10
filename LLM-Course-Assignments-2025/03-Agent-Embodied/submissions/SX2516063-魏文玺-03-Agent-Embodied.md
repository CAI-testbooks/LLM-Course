# 基于LLM的具身智能体系统设计与实现
课程名称： 大模型原理与技术 学生姓名： 魏文玺 学号： SX2516063 日期： 2026年1月10日
## 1. 摘要
本报告详细阐述了一个基于大语言模型的具身智能体的构建过程。项目利用 AI2-THOR 仿真环境，采用了 ReAct (Reasoning + Acting) 框架，实现了智能体在家庭场景下的感知、规划与行动闭环。
针对“寻找并抓取物体”这一复杂任务，系统通过符号化感知将环境信息转化为文本，利用 DeepSeek 进行多步推理，并结合显式记忆模块实现了错误恢复与策略优化。最终部署于 NVIDIA RTX 4090 服务器，
完成了 Demo 视频录制与批量自动化评估。实验结果表明，该智能体具备较高的任务成功率与鲁棒性。

## 2. 实验目的
### 2.1 题目背景

具身智能要求智能体不仅能处理被动数据，还能在物理（或模拟）环境中主动执行任务。本作业旨在构建一个基于 LLM 的智能体，在模拟环境中完成家庭服务任务。

### 2.2 实验要求
| **方面** |     **要求**     |                 **本项目实现方案**                 |
|:------:|:--------------:|:-------------------------------------------:|
|  数据准备  |   使用公开数据集/环境   |              使用 AI2-THOR 仿真环境               |
|  模型选择  | VLA 或 LLM+工具调用 |        采用 LLM + ReAct 范式 (DeepSeek)         |
|  核心功能  |  多步规划、工具使用、恢复  |      实现了 Move/Rotate/Pickup 工具链及错误反馈闭环      |
|  迭代优化  |   策略微调、记忆模块    |            引入任务日志为显式记忆，优化 Prompt            |
|   部署   |   模拟器运行、视频录制   |      在 Linux 服务器部署，通过 OpenCV 合成带思维链的视频      |
|   评估   |     成功率、效率     | 编写 benchmark.py 进行 100 次自动化测试，统计成功率与平均需要的步骤 |


## 3. 实验设计
### 3.1 系统架构
本项目采用模块化设计，主要包含三个核心组件：
#### 3.1.1 感知模块
- 封装 ai2thor 控制器
- 符号化转换： 将视觉图像转换为结构化的文本描述
#### 3.1.2 决策模块
- ReAct 提示词工程： 设计了包含 Thought（思考）和 Action（行动）的 System Prompt
- 上下文管理： 维护对话历史 history，包含用户指令、当前观测及过往的成功/失败记忆
- 闭环反馈： 将执行结果（Success/Fail）及错误信息（ErrorMessage）反馈给下一轮观测
#### 3.1.3 执行模块
- 解析 LLM 输出的文本指令
- 调用模拟器 API 执行动作
- 核心：LLM
### 3.2 关键技术点
#### 3.2.1 错误恢复机制
为了解决智能体“撞墙”或“抓空”的问题，我们在 Prompt 中引入了反馈机制。
- 当动作失败时，环境返回 Action Failed: Object not visible。
- LLM 接收反馈后，生成的下一个 Thought 会是 "I cannot see the apple, I need to explore first"，从而修正行为。
#### 3.2.1 错误恢复机制
在 benchmark.py 中维护了一个 memory_log 列表。
- 记录关键事件（如 "Failed to Pickup Apple"）。
- 在每次询问 LLM 时，将 Summary 注入 Prompt。这防止了智能体在同一个位置死循环。
## 4. 实验设置与实施
### 4.1 实验环境
- 模拟环境：Habitat / AI2-THOR
- 使用模型：Deepseek
- 运行平台：Ubuntu 22.04
- 计算资源：RTX 4090

### 4.2 核心代码片段
**ReAct 提示词设计：**

~~~
SYSTEM_PROMPT = """
You are a smart home robot.
AVAILABLE ACTIONS: MoveAhead(), RotateRight(), Pickup(Object)...
OUTPUT FORMAT:
Thought: <Reasoning>
Action: <Action>
"""
~~~

**主循环设计：**
~~~
obs_text = env.get_observation()
response = llm.chat(history + [obs_text])
action = parse(response)
success = env.step(action)
~~~

## 5. 实验结果与分析
实验结果表明，该具身智能体能够完成指定任务流程，
在多步交互场景中具备一定的鲁棒性。

|       指标       | 结果  | 说明 |
|:--------------:|:---:|:---:|
| Total Episodes | 100 |测试总轮数|
|  Success Rate  | 98% |任务成功率|
|  Avg Steps  | 6.5 |平均完成步数|

利用 OpenCV 录制了任务执行的全过程视频 https://github.com/Atroheim/Agent-Embodied/blob/main/mission_demo.mp4
。
视频包含了实时画面以及智能体的思维过程 (Thought Process)。


## 6. 创新点
- 将 LLM 引入具身智能体的高层规划中
- 采用模块化架构，便于扩展和复现
- 对智能体决策流程进行了工程层面的优化

## 7. 代码与复现说明
代码仓库地址：
https://github.com/Atroheim/Agent-Embodied

代码仓库中提供了完整的环境配置与运行说明，
可根据 README.md 进行复现实验。


