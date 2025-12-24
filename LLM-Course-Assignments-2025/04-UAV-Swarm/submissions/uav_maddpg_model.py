import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 设备配置（CPU即可，无需GPU）
device = torch.device("cpu")


# 1. 策略网络（单个无人机的动作预测）
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 动作归一化到[-1,1]（对应速度/方向）
        )

    def forward(self, state):
        return self.net(state)


# 2. 价值网络（全局视角，评估多智能体动作）
class Critic(nn.Module):
    def __init__(self, total_state_dim, total_action_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(total_state_dim + total_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出Q值
        )

    def forward(self, total_state, total_action):
        x = torch.cat([total_state, total_action], dim=1)
        return self.net(x)


# 3. 简化版MADDPG框架
class SimpleMADDPG:
    def __init__(self, num_uavs=3, state_dim=8, action_dim=3):
        self.num_uavs = num_uavs
        self.state_dim = state_dim  # 单无人机状态：位置(x,y,z)+速度(vx,vy,vz)+最近任务点距离+碰撞风险
        self.action_dim = action_dim  # 动作：x/y/z方向速度调整

        # 初始化每个无人机的Actor和全局Critic
        self.actors = [Actor(state_dim, action_dim).to(device) for _ in range(num_uavs)]
        self.critic = Critic(num_uavs * state_dim, num_uavs * action_dim).to(device)

        # 优化器
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=1e-4) for actor in self.actors]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        # 奖励函数参数
        self.gamma = 0.95  # 折扣因子

    # 单无人机动作预测
    def get_action(self, uav_id, state):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        action = self.actors[uav_id](state)
        return action.detach().numpy()

    # 奖励函数（初始版）
    def calculate_reward(self, uav_states, task_points, collision_flag):
        rewards = []
        for state in uav_states:
            # 状态：[x,y,z,vx,vy,vz,nearest_task_dist,collision_risk]
            nearest_dist = state[6]
            # 基础奖励：靠近任务点+无碰撞
            reward = -0.1 * nearest_dist - 10 * collision_flag
            rewards.append(reward)
        return rewards

    # 迭代优化后的奖励函数（提升覆盖率）
    def calculate_reward_optimized(self, uav_states, task_points, covered_points, collision_flag):
        rewards = []
        total_task = len(task_points)
        coverage_rate = len(covered_points) / total_task
        for state in uav_states:
            nearest_dist = state[6]
            # 新增覆盖率奖励，降低重复巡检惩罚
            reward = -0.1 * nearest_dist - 10 * collision_flag + 5 * coverage_rate - 2 * (state[7] > 0.8)
            rewards.append(reward)
        return rewards

    # 简单训练（单步更新，快速验证）
    def train_step(self, uav_states, uav_actions, rewards, next_uav_states):
        # 转换为张量
        uav_states = [torch.tensor(s, dtype=torch.float32).to(device) for s in uav_states]
        uav_actions = [torch.tensor(a, dtype=torch.float32).to(device) for a in uav_actions]
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_uav_states = [torch.tensor(s, dtype=torch.float32).to(device) for s in next_uav_states]

        # 1. 更新Critic
        total_state = torch.cat(uav_states, dim=0).unsqueeze(0)
        total_action = torch.cat(uav_actions, dim=0).unsqueeze(0)
        q_value = self.critic(total_state, total_action)

        # 计算目标Q值
        next_total_action = torch.cat([self.actors[i](next_uav_states[i]) for i in range(self.num_uavs)],
                                      dim=0).unsqueeze(0)
        next_total_state = torch.cat(next_uav_states, dim=0).unsqueeze(0)
        target_q = rewards.mean() + self.gamma * self.critic(next_total_state, next_total_action)

        # 损失计算
        critic_loss = nn.MSELoss()(q_value, target_q.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 2. 更新Actor
        for i in range(self.num_uavs):
            actor_action = self.actors[i](uav_states[i])
            total_action_i = torch.cat([
                self.actors[j](uav_states[j]) if j != i else actor_action for j in range(self.num_uavs)
            ], dim=0).unsqueeze(0)
            actor_loss = -self.critic(total_state, total_action_i).mean()

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

        return critic_loss.item(), actor_loss.item()


# 初始化模型（供其他文件调用）
if __name__ == "__main__":
    maddpg = SimpleMADDPG(num_uavs=3, state_dim=8, action_dim=3)
    print("MADDPG模型初始化完成")