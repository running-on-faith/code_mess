#! /usr/bin/env python3
"""
@author  : MG
@Time    : 2020/7/4 下午4:31
@File    : __init__.py.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
from dr2.dqn20200209.train.train_drl import train_drl

if __name__ == "__main__":
    epsilon_greedy = 0.1
    gamma = 0.8
    # num_collect_episodes 被默认设置为 epsilon_greedy 倒数的 2 背,以确保又足够的样板,防止由于随机随机策略而导致价值计算失衡
    train_drl(train_loop_count=200,
              num_collect_episodes=int(1 / epsilon_greedy),
              epsilon_greedy=epsilon_greedy,
              train_sample_batch_size=1024,
              train_count_per_loop=50,
              gamma=gamma)
