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
import math
from drl.d3qn_replay_2019_08_25.agent.main import MODEL_NAME, get_agent
from drl.trainer import train_on_each_period
import logging


def train_round_iter_func(round_n_per_target_day, target_avg_holding_days=[4, 5, 7]):
    # 作为训练起始随机动作时，平均换仓天数，该参数可能导致训练后的模型调仓频率变化
    round_n = 1
    for round_n_sub in range(round_n_per_target_day):
        for days in target_avg_holding_days:
            env_kwargs = dict(state_with_flag=True, fee_rate=0.001)
            agent_kwargs = dict(keep_last_action=math.pow(0.5, 1 / days), batch_size=512, epsilon_memory_size=20)
            num_episodes = 2000 + 200 * round_n_sub
            train_kwargs = dict(round_n=round_n, num_episodes=num_episodes, n_episode_pre_record=int(num_episodes / 6),
                                model_name=MODEL_NAME, get_agent_func=get_agent)
            yield round_n, env_kwargs, agent_kwargs, train_kwargs
            round_n += 1


def _test_train_round_iter_func(round_n_per_target_day=3):
    logger = logging.getLogger(__name__)
    for round_n, env_kwargs, agent_kwargs, train_kwargs in train_round_iter_func(round_n_per_target_day):
        logger.error("round_n=%d, env_kwargs=%s, agent_kwargs%s, train_kwargs%s",
                     round_n, env_kwargs, agent_kwargs, train_kwargs)


if __name__ == '__main__':
    train_on_each_period(train_round_iter_func(round_n_per_target_day=2), n_step=60)
    # _test_train_round_iter_func(round_n_per_target_day=2)
