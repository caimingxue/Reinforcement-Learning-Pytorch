#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-23 15:29:24
LastEditor: John
LastEditTime: 2021-04-08 22:36:43
Discription: 
Environment: 
'''
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions import MultivariateNormal
class Actor(nn.Module):
    def __init__(self,state_dim, action_dim, args):
        super(Actor, self).__init__()
        self.has_continuous_action_space = args.has_continuous_action_space
        if self.has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, args.hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hidden_dim, args.hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hidden_dim, action_dim),
                nn.ReLU()
            )
        else:
            self.actor = nn.Sequential(
                    nn.Linear(state_dim, args.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(args.hidden_dim, args.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(args.hidden_dim, action_dim),
                    nn.Softmax(dim=-1)
            )
    def forward(self, state):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            dist = self.actor(state)
            dist = Categorical(dist)
        return dist

class Critic(nn.Module):
    def __init__(self, state_dim,hidden_dim):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
        )
    def forward(self, state):
        value = self.critic(state)
        return value