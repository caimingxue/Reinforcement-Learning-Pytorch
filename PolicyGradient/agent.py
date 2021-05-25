#!/usr/bin/env python
# coding=utf-8
'''
Author: Mingxue Cai
Email: im_caimingxue@163.com
Date: 2021-05-14
Discription:
Environment:
'''
import sys, os
import torch
from torch.distributions import Bernoulli
from torch.distributions import Categorical
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from PolicyGradient.model import MLP


class PolicyGradient:
    def __init__(self, args, state_dim, action_dim):
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.state_dim = state_dim
        self.action_dim = action_dim

        # init N Monte Carlo transitions in one game
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        # init network params
        self.policy_net = MLP(state_dim, action_dim, args.hidden_dim)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=args.lr)

    def choose_action(self, state):
        state = torch.from_numpy(state).float()
        state = Variable(state)
        net_out = self.policy_net(state)
        with torch.no_grad():
            probs_weight = F.softmax(net_out, dim=0).data.numpy()
        action = np.random.choice(range(probs_weight.shape[0]),
                                  p=probs_weight)  #
        # m = Bernoulli(probs)
        # action = m.sample()
        # action = action.data.numpy().astype(int)[0] # convert to scalar
        return action
    def store_transition(self, state, action, reward):
        self.ep_obs.append(state)
        self.ep_as.append(action)
        self.ep_rs.append(reward)

    # 根据一个episode的每个step的reward列表，计算每一个Step的Gt
    def discount_and_norm_rewards(self):
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_cumulative = 0
        for i in reversed(range(0, len(self.ep_rs))):
            # G_t = r_t + γ·r_t+1 + ... = r_t + γ·G_t+1
            running_cumulative = self.ep_rs[i] + self.gamma * running_cumulative
            discounted_ep_rs[i] = running_cumulative
        # normalize episode reward
        reward_mean = np.mean(discounted_ep_rs)
        reward_std = np.std(discounted_ep_rs)
        for i in range(len(discounted_ep_rs)):
            discounted_ep_rs[i] = (discounted_ep_rs[i] - reward_mean)/reward_std
        return discounted_ep_rs
    def learn(self):
        #step 1: discount and normalize episode reward
        # G value for every state
        discounted_ep_rs_norm = self.discount_and_norm_rewards()
        #step 2: gradient asent
        #self.gradient_computation(discounted_ep_rs_norm)
        self.gradient_computation_test(discounted_ep_rs_norm)
        # clear after learn in this episode
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

    def gradient_computation(self, discounted_ep_rs_norm):
        softmax_input = self.policy_net.forward(torch.FloatTensor(self.ep_obs))
        neg_log_prob = F.cross_entropy(input=softmax_input, target=torch.LongTensor(self.ep_as),
                                       reduction="none")
        discounted_ep_rs_norm = torch.FloatTensor(discounted_ep_rs_norm)
        loss = torch.mean(neg_log_prob * discounted_ep_rs_norm)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def gradient_computation_test(self, discounted_ep_rs_norm):
        softmax_input = self.policy_net.forward(torch.FloatTensor(self.ep_obs))
        # all_act_prob = F.softmax(softmax_input, dim=0).detach().numpy()
        neg_log_prob = F.cross_entropy(input=softmax_input, target=torch.LongTensor(self.ep_as),
                                       reduction='none')
        discounted_ep_rs_norm = torch.FloatTensor(discounted_ep_rs_norm)
        loss = torch.mean(neg_log_prob * discounted_ep_rs_norm)
        self.optimizer.zero_grad()
        # probs = self.policy_net(state)
        # m = Bernoulli(probs)
        # # - convert descent to ascent
        # loss = -m.log_prob(action) * reward
        # #print("loss value is :", loss)
        loss.backward()
        self.optimizer.step()
    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path + 'pg_checkpoint.pt')
    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path + 'pg_checkpoint.pt'))




