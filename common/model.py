#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-12 21:14:12
LastEditor: John
LastEditTime: 2021-03-31 13:49:06
Discription: 
Environment: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import math

class MLP(nn.Module):
    def __init__(self, input_dim,output_dim,hidden_dim=128):
        """ 初始化q网络，为全连接网络
            input_dim: 输入的feature即环境的state数目
            output_dim: 输出的action总个数
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim) # 输入层
        self.fc2 = nn.Linear(hidden_dim,hidden_dim) # 隐藏层
        self.fc3 = nn.Linear(hidden_dim, output_dim) # 输出层
        
    def forward(self, x):
        # 各层对应的激活函数
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x


class NoisyDQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(NoisyDQN, self).__init__()

        self.linear = nn.Linear(num_inputs, 256)
        self.noisy1 = NoisyLinear(256, 256)
        self.noisy2 = NoisyLinear(256, num_actions)

    def forward(self, x):
        x = F.relu(self.linear(x))
        x = F.relu(self.noisy1(x))
        x = self.noisy2(x)
        return x

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor([state], device="cpu", dtype=torch.float32)
            q_value = self.forward(state)
            action = q_value.max(1)[1].item()
        return action

    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()

# class ActorCritic(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden_dim):
#         super(ActorCritic, self).__init__()
#         self.critic = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1)
#         )
#
#         self.actor = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, output_dim),
#             # 使得在softmax操作之后在dim这个维度相加等于1
#             # 注意，默认的方法已经弃用，最好在使用的时候声明dim
#             nn.Softmax()
#         )
#
#     def forward(self, x):
#         # critic: evaluates value in the state s_t
#         value = self.critic(x)
#         # actor: choses action to take from state s_t
#         # by returning probability of each action
#         probs = self.actor(x)
#
#         # 分类,对actor输出的动作概率进行分类统计
#         # create a categorical distribution over the list of probabilities of actions
#         dist  = Categorical(probs)
#
#         # return values for both actor and critic as a tuple of 2 values:
#         # 1(action prob). a list with the probability of each action over the action space
#         # 2(state value). the value from state s_t
#         return dist, value

class actor(nn.Module):  # policy net
    # actor: choses action to take from state s_t
    # by returning probability of each action
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        prob = self.softmax(x)
        # 分类,对actor输出的动作概率进行分类统计
        # create a categorical distribution over the list of probabilities of actions
        dist = Categorical(prob)
        return dist

class critic(nn.Module):  # Q net:evaluates value in the state s_t
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class baseline_net(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(baseline_net,self).__init__()
        self.fc1=nn.Linear(input_dim, hidden_dim)
        self.fc2=nn.Linear(hidden_dim, output_dim)
    def forward(self,x):
        x=self.fc1(x)
        x=F.relu(x)
        out=self.fc2(x)
        return out