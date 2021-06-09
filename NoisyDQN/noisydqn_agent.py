import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import math
import numpy as np
from common.model import NoisyDQN
from common.replay_buffer import PrioritizedReplayBuffer

class NOISY_DQN:
    def __init__(self, state_dim, action_dim, args):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = args.hidden_dim
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.gamma = args.gamma
        self.double_DQN = args.double_DQN
        # e-greedy策略相关参数
        self.frame_idx = 0  # 用于epsilon的衰减计数
        self.epsilon = lambda frame_idx: args.epsilon_end + \
                        (args.epsilon_start - args.epsilon_end) * \
                        math.exp(-1. * frame_idx / args.epsilon_decay)
        self.replay_buffer_capacity = args.replay_buffer_capacity
        self.device = args.device

        self.policy_net = NoisyDQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net = NoisyDQN(self.state_dim, self.action_dim).to(self.device)
        # 加载之前训练好的模型，没有预训练好的模型时可以注释
        # model_path = "/Users/cmx/github_project/Reinforcement-Learning-Pytorch/DQN/saved_model/20210604-151247/pg_checkpoint.pt"
        # self.policy_net.load_state_dict(torch.load(model_path))
        #Synchronize policy net and target net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        #self.target_net = copy.deepcopy(self.policy_net)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        beta_start = 0.4
        beta_frames = 1000
        self.beta_by_frame = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)
        self.replay_buffer = PrioritizedReplayBuffer(self.replay_buffer_capacity, alpha=0.6)

    def choose_action(self, state):
        '''choose action
        '''
        self.frame_idx += 1
        if random.random() > self.epsilon(self.frame_idx):
            with torch.no_grad():
                state = torch.tensor([state], device=self.device, dtype=torch.float32)
                q_value = self.policy_net(state) #state-action value . the net use theta value
                action = q_value.max(1)[1].item() #返回索引值
        else:
            action = random.randrange(self.action_dim)
        return action
    def act(self, state):
        action = self.policy_net.act(state)
        return action

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        beta = self.beta_by_frame(self.frame_idx)

        # get the transition data from replay_buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, \
        weights_batch, indicues_batch = self.replay_buffer.sample(self.batch_size, beta)
        # trans to tensor type
        state_batch = torch.tensor(state_batch, device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float)
        next_state_batch = torch.tensor(next_state_batch, device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device)
        weights_batch = torch.FloatTensor(weights_batch)
        '''计算当前(s_t,a)对应的Q(s_t, a)'''
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        q_values_temp = self.policy_net(state_batch)
        q_values = q_values_temp.gather(dim=1, index=action_batch).squeeze(1) # state-action value
        #the net use theta- value
        if self.double_DQN:
            next_state_actions = self.policy_net(next_state_batch).max(1)[1]
            next_q_values = self.target_net(next_state_batch).gather(dim=1, index=next_state_actions.unsqueeze(-1)).squeeze(-1)
        else:
            next_q_values = self.policy_net(next_state_batch).max(1)[0].detach()  # detach用于冻结梯度，防止对target_Q进行更新。不参与反向传播计算

        #compute expected q_values （V（s+1））if s is final state in an episode, next_q_value is NULL(done batch ==1)
        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        # self.loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))
        self.loss = (q_values - expected_q_values.detach()).pow(2) * weights_batch
        prios = self.loss + 1e-5
        self.loss = self.loss.mean()
        # optimize model
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        prios_trans = prios.data.cpu().numpy()
        self.replay_buffer.update_priorities(indicues_batch, prios_trans)
        self.policy_net.reset_noise()
        self.target_net.reset_noise()
    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path + 'pg_checkpoint.pt')

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path + 'pg_checkpoint.pt'))







