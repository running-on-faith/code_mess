#! /usr/bin/env python3
"""
@author  : MG
@Time    : 2020/8/23 下午2:53
@File    : train_20200823_base_on_20200623_517fb234.py
@contact : mmmaaaggg@163.com
@desc    :
"""
from dr2.dqn20200209.train.train_drl import train_drl

if __name__ == "__main__":
    train_drl(train_loop_count=150, num_collect_episodes=5)
