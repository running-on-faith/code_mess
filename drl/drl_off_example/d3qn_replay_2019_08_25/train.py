#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 2019/9/2 20:54
@File    : off_example.py
@contact : mmmaaaggg@163.com
@desc    : 用于进行指定日期范围数据训练，以及样本外测试
"""
# if True:
#     from ibats_common.backend.rl.utils import use_cup_only
#
#     use_cup_only()

from drl.trainer import train_on_each_period
from drl.d3qn_replay_2019_08_25.agent.main import MODEL_NAME, get_agent

if __name__ == '__main__':
    train_on_each_period(model_name=MODEL_NAME, get_agent_func=get_agent, round_from=0, round_max=4)
