#! /usr/bin/env python3
"""
@author  : MG
@Time    : 2020/8/24 下午8:25
@File    : train_20200824_fac38f3d_reproduce_20200623_517fb234.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
from dr2.dqn20200209.train.train_drl import train_drl

if __name__ == "__main__":
    train_drl(train_loop_count=500, num_collect_episodes=5)
