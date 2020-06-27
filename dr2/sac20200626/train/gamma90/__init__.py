#! /usr/bin/env python3
"""
@author  : MG
@Time    : 2020/6/27 下午6:35
@File    : __init__.py.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
from dr2.sac20200626.train.train_drl import train_drl

train_drl(train_loop_count=300, num_collect_episodes=10, agent_kwargs={"gamma": 0.9})

if __name__ == "__main__":
    pass
