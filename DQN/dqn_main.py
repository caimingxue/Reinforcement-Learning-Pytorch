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
from common.utils import create_file, save_results
from common.plot import plot_rewards
from dqn_agent import DQN
from args_config import get_args


curr_path = os.path.dirname(__file__)
# parent_path = os.path.dirname(curr_path)
# sys.path.append(parent_path)  # add current terminal path to sys.path
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
saved_model_path, result_path = create_file(curr_time, curr_path)

Transitions = collections.namedtuple('Transitions', field_names=['state', 'action', 'reward', 'next_state', 'done'])


def train(args, env, agent):
    print('Start to train !')
    print(f'Env:{args.env_name}, Algorithm:{args.algo}, Device:{args.device}')
    rewards = []
    ma_rewards = []  # moveing average reward
    t0 = time.time()
    for i_episode in range(args.train_eps):
        state = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            transition = Transitions(state, action, reward, next_state, done)
            agent.replay_buffer.push(transition)
            state = next_state
            agent.learn()
            env.render()
        print('Training  | Episode: {}/{}  | Episode Reward: {:.0f}  | Running Time: {:.4f}'.format(
            i_episode + 1, args.train_eps, ep_reward, time.time() - t0
        )
        )
        # target net update its params by policy net
        if i_episode % args.target_network_update_frequency == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        rewards.append(ep_reward)
        # 计算滑动窗口的reward
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print('Complete training！')
    return rewards, ma_rewards

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
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQN(state_dim, action_dim, args)
    rewards, ma_rewards = train(args, env, agent)
    time_end = time.time()
    total_time = time_end - time_start
    print("The total time is", total_time)
    agent.save_model(saved_model_path)
    save_results(result_path, ma_rewards, tag='train', path=result_path)
    plot_rewards(rewards, ma_rewards, tag="train", algo="DQN", path=result_path)

    agent.load_model(saved_model_path)
    eval_model(args.env_name, agent, args.seed, args.test_eps)