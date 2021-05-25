#!/usr/bin/env python
# coding=utf-8
'''
Author: Mingxue Cai
Email: im_caimingxue@163.com
Date: 2021-05-14
Discription:
Environment:
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    '''
    多层感知机
    input：state dim
    output: action probility
    '''
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim) # output is an action prob
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x