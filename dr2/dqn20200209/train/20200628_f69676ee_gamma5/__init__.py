#! /usr/bin/env python3
"""
@author  : MG
@Time    : 2020/6/28 下午10:18
@File    : __init__.py.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
from dr2.dqn20200209.train.train_drl import train_drl

if __name__ == "__main__":
    train_drl(train_loop_count=200, num_collect_episodes=10, train_count_per_loop=50, gamma=0.5)