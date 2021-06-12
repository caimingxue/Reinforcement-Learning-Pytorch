import random

import torch
import torch.optim as optim
import numpy as np
from collections import namedtuple
from common.model import actor, critic

class AC:
    def __init__(self, state_dim, action_dim, args):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = args.hidden_dim
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.gamma = args.gamma

        self.returns = []
        self.log_prob_buffet = []
        self.state_value_buffer = []
        self.entropy = 0.0

        self.device = args.device

        self.actor_net = actor(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.critic_net = critic(self.state_dim, 1, self.hidden_dim).to(self.device)
        # self.target_net = ActorCritic(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        # 加载之前训练好的模型，没有预训练好的模型时可以注释
        # model_path = "/Users/cmx/github_project/Reinforcement-Learning-Pytorch/DQN/saved_model/20210604-151247/pg_checkpoint.pt"
        # self.policy_net.load_state_dict(torch.load(model_path))
        #Synchronize policy net and target net
        self.target_net.load_state_dict(self.critic_net.state_dict())
        self.target_net.eval()
        #self.target_net = copy.deepcopy(self.policy_net)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.lr)

        SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

    def choose_action(self, state):
        '''choose action
        ## 选择动作，这个动作不是根据Q值来选择，而是使用softmax生成的概率来选
        ## 不需要epsilon-greedy，因为概率本身就具有随机性
        '''
        state = torch.FloatTensor(state).to(self.device)
        action_dist = self.actor_net(state)
        state_value = self.critic_net(state)

        #sample an action using the distribution
        action = action_dist.sample()

        # 取对数似然 logπ(s,a),这里就等于action的one-hot*action的概率
        # m.log_prob(action)相当于probs.log()[0][action.item()].unsqueeze(0)
        self.log_prob = action_dist.log_prob(action)
        self.log_prob_buffet.append(self.log_prob)
        self.state_value_buffer.append(state_value)

        self.entropy = self.entropy + action_dist.entropy().mean()

        # the action to take (left or right)
        return action.item()

    def learn(self):
        """
            Training code. Calculates actor and critic loss and performs backprop.
            """
        # 计算loss均值
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss - 0.001 * self.entropy

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        # reset rewards and action buffer
        del self.log_prob_buffet[:]
        del self.state_value_buffer[:]

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path + 'pg_checkpoint.pt')

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path + 'pg_checkpoint.pt'))

