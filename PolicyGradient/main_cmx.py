#!/usr/bin/env python
# coding=utf-8
'''
Author: Mingxue Cai
Email: im_caimingxue@163.com
Date: 2021-05-14
Discription:
Environment:
'''
import sys,os
import time
sys.path.append(os.getcwd()) # add current terminal path to sys.path
import torch
import torch.nn as nn
import gym
import matplotlib.pyplot as plt
import datetime
import argparse
from itertools import count
from PolicyGradient.agent_cmx import PolicyGradient
from common.plot import plot_rewards
from common.utils import save_results

SEQUENCE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # obtain current time
SAVED_MODEL_PATH = os.path.split(os.path.abspath(__file__))[0]+"/saved_model/"+SEQUENCE+'/'  # path to save model
if not os.path.exists(os.path.split(os.path.abspath(__file__))[0]+"/saved_model/"):
    os.mkdir(os.path.split(os.path.abspath(__file__))[0]+"/saved_model/")
if not os.path.exists(SAVED_MODEL_PATH):
    os.mkdir(SAVED_MODEL_PATH)
RESULT_PATH = os.path.split(os.path.abspath(__file__))[0]+"/results/"+SEQUENCE+'/' # path to save rewards
if not os.path.exists(os.path.split(os.path.abspath(__file__))[0]+"/results/"):
    os.mkdir(os.path.split(os.path.abspath(__file__))[0]+"/results/")
if not os.path.exists(RESULT_PATH):
    os.mkdir(RESULT_PATH)

parser = argparse.ArgumentParser(description='Pytorch REINFORCE example')
parser.add_argument("--env_name", default="CartPole-v0")
parser.add_argument("--train_eps", type=int, default=600)
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',help='discount factor(default:0.99)')
parser.add_argument('--hidden_dim', type=int, default=36)
parser.add_argument('--seed',type=int, default=10, metavar='N',help='random seed (default: 543)')
parser.add_argument('--render',action='store_false',help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--train', dest='train', action='store_true', default=True)
args = parser.parse_args()

def train(args, env, agent):
    ''' 存储每个episode的reward用于绘图'''
    rewards_epsisode = []
    ma_rewards_epsisode = []
    t0 = time.time()
    for episode in range(args.train_eps):
        state = env.reset()
        total_reward = 0
        for _ in count():
            # env.render()
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward)
            total_reward += reward
            state = next_state
            if done:
                break
        agent.learn()
        print('Training  | Episode: {}/{}  | Episode Reward: {:.0f}  | Running Time: {:.4f}'.format(
                episode + 1, args.train_eps, total_reward, time.time() - t0
                )
        )
        rewards_epsisode.append(total_reward)
        if ma_rewards_epsisode:
            ma_rewards_epsisode.append(0.9 * ma_rewards_epsisode[-1] + 0.1 * total_reward)
        else:
            ma_rewards_epsisode.append(total_reward)

        if episode % args.batch_size == 0 and episode > 0:

            print("***************")
    print("complete training")
    return rewards_epsisode, ma_rewards_epsisode

def eval_model(env_name, learned_agent, seed, eval_episodes):
    eval_env = gym.make(env_name)
    eval_env.seed(args.seed)
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
    env = gym.make(args.env_name)
    env.seed(args.seed)
    torch.manual_seed(args.seed)   # 策略梯度算法方差很大，设置seed以保证复现性
    state_dim = env.observation_space.shape[0]
    print('observation space:', state_dim)
    action_dim = env.action_space.n
    print('action space:', action_dim)
    reward_threshold_episode = env.spec.reward_threshold
    print("reward_threshold_episode", reward_threshold_episode)
    agent = PolicyGradient(args, state_dim, action_dim)
    rewards_epsisode, ma_rewards_epsisode = train(args, env, agent)
    time_end = time.time()
    total_time = time_end - time_start
    print("The total time is", total_time)
    agent.save_model(SAVED_MODEL_PATH)
    save_results(rewards_epsisode, ma_rewards_epsisode, tag='train', path=RESULT_PATH)
    plot_rewards(rewards_epsisode, ma_rewards_epsisode, tag="train", algo="Policy Gradient", path=RESULT_PATH)
    if args.train:
        agent.load_model(SAVED_MODEL_PATH)
        eval_model(args.env_name, agent, 1, 50)
