#!/usr/bin/env python
# coding=utf-8
'''
Author: Mingxue Cai
Email: im_caimingxue@163.com
Date: 2021-05-24
Environment: 
'''
import argparse
import sys,os
curr_path = os.path.dirname(__file__)
parent_path=os.path.dirname(curr_path) 
sys.path.append(parent_path) # add current terminal path to sys.path
import gym
import numpy as np
import torch
import datetime
from PPO.agent_ppo import PPO
from common.plot import plot_rewards
from common.utils import save_results,make_dir
import parser
import time

CURRENT_TIME = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # obtain current time
SAVED_MODEL_PATH = os.path.split(os.path.abspath(__file__))[0]+"/saved_model/"+CURRENT_TIME+'/'  # path to save model
if not os.path.exists(os.path.split(os.path.abspath(__file__))[0]+"/saved_model/"):
    os.mkdir(os.path.split(os.path.abspath(__file__))[0]+"/saved_model/")
if not os.path.exists(SAVED_MODEL_PATH):
    os.mkdir(SAVED_MODEL_PATH)
RESULT_PATH = os.path.split(os.path.abspath(__file__))[0]+"/results/"+CURRENT_TIME+'/' # path to save rewards
if not os.path.exists(os.path.split(os.path.abspath(__file__))[0]+"/results/"):
    os.mkdir(os.path.split(os.path.abspath(__file__))[0]+"/results/")
if not os.path.exists(RESULT_PATH):
    os.mkdir(RESULT_PATH)

parser = argparse.ArgumentParser(description='Pytorch PPO')
parser.add_argument("--env_name", default="CartPole-v0")
parser.add_argument('--seed',type=int, default=10, metavar='N', help='random seed (default: 543)')
parser.add_argument("--train_eps", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=5)
parser.add_argument("--actor_lr", type=float, default=0.0003)
parser.add_argument("--critic_lr", type=float, default=0.0003)
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='reward discount in TD error -- Critic')
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--policy_clip", type=float, default=0.2)
parser.add_argument("--update_fre", type=int, default=20, help="frequency of agent update")
parser.add_argument("--n_epochs", type=int, default=4)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument("--device", default="cpu")
parser.add_argument('--test', dest='test', action='store_true', default=True)
parser.add_argument('--has_continuous_action_space', action='store_false', default=False)
parser.add_argument("--test_eps", type=int, default=60)

args = parser.parse_args()
        
def train(args, env, agent):
    rewards= []
    ma_rewards = [] # moving average rewards
    running_step = 0
    for episode in range(args.train_eps): # in one episode
        state = env.reset()
        ep_reward = 0
        t0 = time.time()
        done = False
        while not done:
            action, prob, val = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            agent.memory.push(state, action, prob, val, reward, done)
            state = next_state
            # N步更新的方法，每update_fre步了就可以进行一次更新
            running_step += 1
            if running_step % args.update_fre == 0 or episode == args.train_eps - 1:
                agent.learn()

        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(
                0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
        print('Training  | Episode: {}/{}  | Episode Reward: {:.0f}  | Running Time: {:.4f}'.format(
            episode + 1, args.train_eps, ep_reward, time.time() - t0
        )
        )
    return rewards,ma_rewards

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
            action, prob, val = learned_agent.choose_action(state)
            state, reward, done, _ = eval_env.step(action)
            total_reward += reward
        print('Testing  | Episode: {}/{}  | Episode Reward: {:.0f}  | Running Time: {:.4f}'.format(
                episode + 1, eval_episodes, total_reward, time.time() - t0
                )
        )
        rewards_epsisode.append(total_reward)
if __name__ == '__main__':
    env = gym.make(args.env_name)
    env.seed(args.seed) # Set seeds
    state_dim=env.observation_space.shape[0]
    action_dim=env.action_space.n
    agent = PPO(state_dim, action_dim, args)
    rewards,ma_rewards = train(args, env, agent)
    agent.save_model(SAVED_MODEL_PATH)
    save_results(rewards, ma_rewards,tag='train', path=RESULT_PATH)
    plot_rewards(rewards, ma_rewards,tag="train", env=args.env_name, algo = "PPO", path=RESULT_PATH)

    if args.test:
        agent.load_model(SAVED_MODEL_PATH)
        eval_model(args.env_name, agent, args.seed, args.test_eps)