#!/usr/bin/env python
# coding=utf-8
'''
Author: Mingxue Cai
Email: im_caimingxue@163.com
Date: 2021-05-25
Environment: 
'''
import os
import numpy as np
import torch 
import torch.optim as optim
from PPO.model import Actor, Critic
from PPO.memory import PPOMemory
class PPO:
    def __init__(self, state_dim, action_dim, args):
        self.gamma = args.gamma
        self.policy_clip = args.policy_clip
        self.n_epochs = args.n_epochs
        self.gae_lambda = args.gae_lambda
        self.device = args.device
        self.actor = Actor(state_dim, action_dim, args).to(self.device)
        self.critic = Critic(state_dim, args.hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        self.memory = PPOMemory(args.batch_size)
        self.loss = 0

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(self.device)#注意cuda上面的变量类型只能是tensor，不能是其他
        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()
        probs = torch.squeeze(dist.log_prob(action)).item()#log_prob(value)是计算value在定义的正态分布（mean,1）中对应的概率的对数，正太分布概率密度函数是
        action = torch.squeeze(action).item()#item()返回的是tensor中的值，且只能返回单个值（标量），不能返回向量，使用返回loss等
        value = torch.squeeze(value).item()
        return action, probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.sample()
            values = vals_arr

            advantage = self.compute_advantages(reward_arr, dones_arr, values)

            ### SGD ###
            values = torch.tensor(values).to(self.device)
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.device)
                actions = torch.tensor(action_arr[batch]).to(self.device)
                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = torch.squeeze(critic_value)
                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()
                total_loss = actor_loss + 0.5*critic_loss
                self.loss  = total_loss
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        self.memory.clear()
    def compute_advantages(self, reward_arr, dones_arr, values):
        ### compute advantage ###
        advantage = np.zeros(len(reward_arr), dtype=np.float32)
        for t in range(len(reward_arr) - 1):
            discount = 1
            a_t = 0
            for k in range(t, len(reward_arr) - 1):
                a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * \
                                   (1 - int(dones_arr[k])) - values[k])
                discount *= self.gamma * self.gae_lambda
            advantage[t] = a_t
        advantage = torch.tensor(advantage).to(self.device)
        return advantage

    def compute_rtgs(self, batch_rews):
        """
            Compute the Reward-To-Go of each timestep in a batch given the rewards.

            Parameters:
                batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

            Return:
                batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
        """
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []

        # Iterate through each episode
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0  # The discounted reward so far

            # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
            # discounted return (think about why it would be harder starting from the beginning)
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs

    def save_model(self,path):
        actor_checkpoint = os.path.join(path, 'ppo_actor.pt')
        critic_checkpoint= os.path.join(path, 'ppo_critic.pt')
        torch.save(self.actor.state_dict(), actor_checkpoint)
        torch.save(self.critic.state_dict(), critic_checkpoint)
    def load_model(self,path):
        actor_checkpoint = os.path.join(path, 'ppo_actor.pt')
        critic_checkpoint= os.path.join(path, 'ppo_critic.pt')
        self.actor.load_state_dict(torch.load(actor_checkpoint)) 
        self.critic.load_state_dict(torch.load(critic_checkpoint))  


