"""
@ Author: Peter Xiao
@ Date: 2020.7.20
@ Filename: PG.py
@ Brief: 使用 蒙特卡洛策略梯度Reinforce训练CartPole-v0
"""

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
from collections import deque

# Hyper Parameters for PG Network
GAMMA = 0.95  # discount factor
LR = 0.01  # learning rate

# torch.backends.cudnn.enabled = False  # 非确定性算法


class PGNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PGNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 20)
        self.fc2 = nn.Linear(20, action_dim)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            nn.init.normal_(m.weight.data, 0, 0.1)
            nn.init.constant_(m.bias.data, 0.01)
            # m.bias.data.zero_()


class PG(object):
    # dqn Agent
    def __init__(self, env):  # 初始化
        # 状态空间和动作空间的维度
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n


        # init N Monte Carlo transitions in one game
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        # init network parameters
        self.network = PGNetwork(state_dim=self.state_dim, action_dim=self.action_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)

        # init some parameters
        self.time_step = 0

    def choose_action(self, observation):
        observation = torch.FloatTensor(observation)
        network_output = self.network.forward(observation)
        with torch.no_grad():
            prob_weights = F.softmax(network_output, dim=0).data.numpy()
        # prob_weights = F.softmax(network_output, dim=0).detach().numpy()
        action = np.random.choice(range(prob_weights.shape[0]),
                                  p=prob_weights)  # select action w.r.t the actions prob
        return action

    # 将状态，动作，奖励这一个transition保存到三个列表中
    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        self.time_step += 1

        # Step 1: 计算每一步的状态价值
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        # 注意这里是从后往前算的，所以式子还不太一样。算出每一步的状态价值
        # 前面的价值的计算可以利用后面的价值作为中间结果，简化计算；从前往后也可以
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * GAMMA + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        discounted_ep_rs -= np.mean(discounted_ep_rs)  # 减均值
        discounted_ep_rs /= np.std(discounted_ep_rs)  # 除以标准差
        discounted_ep_rs = torch.FloatTensor(discounted_ep_rs)

        # Step 2: 前向传播
        softmax_input = self.network.forward(torch.FloatTensor(self.ep_obs))
        # all_act_prob = F.softmax(softmax_input, dim=0).detach().numpy()
        neg_log_prob = F.cross_entropy(input=softmax_input, target=torch.LongTensor(self.ep_as),
                                       reduction='none')

        # Step 3: 反向传播
        loss = torch.mean(neg_log_prob * discounted_ep_rs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 每次学习完后清空数组
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []


# ---------------------------------------------------------
# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE = 1000  # Episode limitation
STEP = 300  # Step limitation in an episode
TEST = 10  # The number of experiment test every 100 episode


def main():
    # initialize OpenAI Gym env and dqn agent
    env = gym.make(ENV_NAME)
    env.seed(1)
    torch.manual_seed(1)   # 策略梯度算法方差很大，设置seed以保证复现性
    agent = PG(env)

    for episode in range(EPISODE):
        # initialize task
        state = env.reset()
        # Train
        # 只采一盘？N个完整序列
        for step in range(STEP):
            action = agent.choose_action(state)  # softmax概率选择action
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward)  # 新函数 存取这个transition
            state = next_state
            if done:
                # print("stick for ",step, " steps")
                agent.learn()  # 更新策略网络
                break

        # Test every 100 episodes
        if episode % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEP):
                    env.render()
                    action = agent.choose_action(state)  # direct action for test
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward / TEST
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)


if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    print('The total time is ', time_end - time_start)