#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-12 16:02:24
LastEditor: John
LastEditTime: 2021-04-13 18:34:20
Discription: 
Environment: 
'''
import os, sys
# sys.path.append(os.getcwd()) # add current terminal path to sys.path
import numpy as np
from pathlib import Path

def create_file(current_datetime, current_path):

    SAVED_MODEL_PATH = current_path + "/saved_model/" + current_datetime + '/'  # path to save model
    if not os.path.exists(current_path + "/saved_model/"):
        os.mkdir(current_path + "/saved_model/")
    if not os.path.exists(SAVED_MODEL_PATH):
        os.mkdir(SAVED_MODEL_PATH)

    RESULT_PATH = current_path + "/results/" + current_datetime + '/'  # path to save rewards
    if not os.path.exists(current_path + "/results/"):
        os.mkdir(current_path + "/results/")
    if not os.path.exists(RESULT_PATH):
        os.mkdir(RESULT_PATH)
    return SAVED_MODEL_PATH, RESULT_PATH

def save_results(rewards,ma_rewards,tag='train',path='./results'):
    '''保存reward等结果
    '''
    np.save(path+'rewards_'+tag+'.npy', rewards)
    np.save(path+'ma_rewards_'+tag+'.npy', ma_rewards)
    print('results saved!')

def make_dir(*paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
def del_empty_dir(*paths):
    '''del_empty_dir delete empty folders unders "paths"
    '''
    for path in paths:
        dirs = os.listdir(path)
        for dir in dirs:
            if not os.listdir(os.path.join(path, dir)):
                os.removedirs(os.path.join(path, dir))