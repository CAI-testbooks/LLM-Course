import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = torch.device("cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.net(state)


class Critic(nn.Module):
    def __init__(self, total_state_dim, total_action_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(total_state_dim + total_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, total_state, total_action):
        x = torch.cat([total_state, total_action], dim=1)
        return self.net(x)


class SimpleMADDPG:
    def __init__(self, num_uavs=3, state_dim=8, action_dim=3):
        self.num_uavs = num_uavs
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actors = [Actor(state_dim, action_dim).to(device) for _ in range(num_uavs)]
        self.critic = Critic(num_uavs * state_dim, num_uavs * action_dim).to(device)

        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=1e-4) for actor in self.actors]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.gamma = 0.95
        self.covered_history = set()  # 记录历史覆盖的任务点

    def get_action(self, uav_id, state):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        action = self.actors[uav_id](state)
        return action.detach().numpy()

    def calculate_reward_optimized(self, uav_states, task_points, covered_points, collision_flag, assigned_tasks):
        """
        优化的奖励函数：
        - 增加新覆盖任务点的奖励
        - 提高碰撞惩罚
        - 增加任务进度奖励
        """
        rewards = []
        total_task = len(task_points)
        new_covered = covered_points - self.covered_history
        self.covered_history = covered_points.copy()

        for i, state in enumerate(uav_states):
            nearest_dist = state[6]
            collision_risk = state[7]

            # 基础奖励：靠近任务点
            reward = -0.15 * nearest_dist  # 增强距离惩罚

            # 新覆盖任务点奖励（每个新点+20）
            uav_new_covered = 0
            if assigned_tasks[i]:
                uav_new_covered = sum(1 for t in assigned_tasks[i] if t in new_covered)
            reward += 20 * uav_new_covered

            # 碰撞惩罚（更严厉）
            reward -= 30 * collision_flag  # 从10提高到30
            reward -= 5 * collision_risk  # 增加风险惩罚

            # 任务进度奖励
            coverage_rate = len(covered_points) / total_task
            reward += 10 * coverage_rate  # 从5提高到10

            rewards.append(reward)

        return rewards

    def train_step(self, uav_states, uav_actions, rewards, next_uav_states):
        uav_states = [torch.tensor(s, dtype=torch.float32).to(device) for s in uav_states]
        uav_actions = [torch.tensor(a, dtype=torch.float32).to(device) for a in uav_actions]
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_uav_states = [torch.tensor(s, dtype=torch.float32).to(device) for s in next_uav_states]

        # 更新Critic
        total_state = torch.cat(uav_states, dim=0).unsqueeze(0)
        total_action = torch.cat(uav_actions, dim=0).unsqueeze(0)
        q_value = self.critic(total_state, total_action)

        next_total_action = torch.cat([self.actors[i](next_uav_states[i]) for i in range(self.num_uavs)],
                                      dim=0).unsqueeze(0)
        next_total_state = torch.cat(next_uav_states, dim=0).unsqueeze(0)
        target_q = rewards.mean() + self.gamma * self.critic(next_total_state, next_total_action)

        critic_loss = nn.MSELoss()(q_value, target_q.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)  # 保留计算图
        self.critic_optimizer.step()

        # 更新Actor（多轮更新提高学习效率）
        actor_losses = []
        for i in range(self.num_uavs):
            actor_action = self.actors[i](uav_states[i])
            total_action_i = torch.cat([
                self.actors[j](uav_states[j]) if j != i else actor_action for j in range(self.num_uavs)
            ], dim=0).unsqueeze(0)
            actor_loss = -self.critic(total_state, total_action_i).mean()
            actor_losses.append(actor_loss.item())

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward(retain_graph=True)  # 保留计算图
            self.actor_optimizers[i].step()

        return critic_loss.item(), np.mean(actor_losses)


if __name__ == "__main__":
    maddpg = SimpleMADDPG(num_uavs=3, state_dim=8, action_dim=3)
    print("MADDPG模型初始化完成")