#!/usr/bin/env python
# coding=utf-8
'''
@Author: Mingxue Cai
@Email: im_caimingxue@163.com
@Date: 2021-06-03 15:00
'''
'''off-policy
'''
import sys,os
import datetime
import gym
import collections
import time
import torch
from common.utils import create_file, save_results
from common.plot import plot_rewards
from args_config import get_args
from AC.ac_agent import AC


curr_path = os.path.dirname(__file__)
# parent_path = os.path.dirname(curr_path)
# sys.path.append(parent_path)  # add current terminal path to sys.path
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
saved_model_path, result_path = create_file(curr_time, curr_path)

Transitions = collections.namedtuple('Transitions', field_names=['state', 'action', 'reward', 'next_state', 'done'])


def train(args, env, agent):
    print('Start to train !')
    print(f'Env:{args.env_name}, Algorithm:{args.algo}, Device:{args.device}')
    ma_rewards = []  # moveing average reward
    t0 = time.time()
    for i_episode in range(args.train_eps):
        state = env.reset()
        done = False
        ep_reward = 0
        num_step = 0
        done_mask = []
        rewards = []
        while not done:
            env.render()
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            rewards.append(reward)
            done_mask.append(1-done)

            state = torch.FloatTensor(state).to(args.device)
            state_value_eval = agent.critic_net(state)
            next_state = torch.FloatTensor(next_state).to(args.device)
            # use target net to compute value of next state to avoid "自举"
            TD_target = reward + args.gamma * agent.target_net(next_state)
            # TD_error = state_value_eval - TD_target.detach()
            TD_error = TD_target.detach() - state_value_eval
            critic_net_loss = TD_error ** 2
            agent.critic_optimizer.zero_grad()
            critic_net_loss.backward()
            agent.critic_optimizer.step()

            # actor_net_loss = -(agent.log_prob * TD_error.detach())
            actor_net_loss = -(agent.log_prob * state_value_eval.detach())
            agent.critic_optimizer.zero_grad()
            actor_net_loss.backward()
            agent.critic_optimizer.step()

            state = next_state
            num_step += 1
        agent.target_net.load_state_dict(agent.critic_net.state_dict())


        # rewards = torch.tensor(rewards, device=args.device, dtype=torch.float).unsqueeze(1)
        # done_mask.append(torch.FloatTensor(1 - done).unsqueeze(1).to(args.device))
        #
        # next_state = torch.FloatTensor(next_state).to(args.device)
        # next_state_value = agent.critic_net(next_state)
        #
        # agent.returns = compute_returns(next_state_value, rewards, done_mask)
        # agent.learn()

        print('Training  | Episode: {}/{}  | Episode Reward: {:.0f}  | Running Time: {:.4f}'.format(
            i_episode + 1, args.train_eps, ep_reward, time.time() - t0
        )
        )
        # target net update its params by policy net
        # rewards.append(ep_reward)
        # 计算滑动窗口的reward
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print('Complete training！')
    # return rewards, ma_rewards

def compute_returns(next_value, rewards, done_masks, gamma=0.99):
        """
        # calculate the true value using rewards returned from the environment
        """
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * (done_masks[step])
            # list.insert(index, obj)，index -- 对象 obj 需要插入的索引位置。
            returns.insert(0, R)
        return returns

def eval_model(env_name, learned_agent, seed, eval_episodes):
    eval_env = gym.make(env_name)
    eval_env.seed(seed)
    rewards_epsisode, ma_rewards_epsisode = [], []
    t0 = time.time()
    for episode in range(eval_episodes):
        total_reward = 0
        state, done = eval_env.reset(), False
        while not done:
            eval_env.render()
            action = learned_agent.choose_action(state)
            state, reward, done, _ = eval_env.step(action)
            total_reward += reward
        print('Testing  | Episode: {}/{}  | Episode Reward: {:.0f}  | Running Time: {:.4f}'.format(
                episode + 1, eval_episodes, total_reward, time.time() - t0
                )
        )
        rewards_epsisode.append(total_reward)

if __name__ == "__main__":
    time_start = time.time()
    args = get_args()
    env = gym.make(args.env_name)
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = AC(state_dim, action_dim, args)
    rewards, ma_rewards = train(args, env, agent)
    time_end = time.time()
    total_time = time_end - time_start
    print("The total time is", total_time)
    agent.save_model(saved_model_path)
    save_results(result_path, ma_rewards, tag='train', path=result_path)
    plot_rewards(rewards, ma_rewards, tag="train", algo="DQN", path=result_path)

    agent.load_model(saved_model_path)
    eval_model(args.env_name, agent, args.seed, args.test_eps)