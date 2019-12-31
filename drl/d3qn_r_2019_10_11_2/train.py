#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 2019/9/2 20:54
@File    : off_example.py
@contact : mmmaaaggg@163.com
@desc    : 用于进行指定日期范围数据训练，以及样本外测试
"""
import logging
# if True:
#     from ibats_common.backend.rl.utils import use_cup_only
#
#     use_cup_only()
import math

from drl import DATA_FOLDER_PATH
from drl.d3qn_r_2019_10_11_2.agent.main import MODEL_NAME, get_agent
from drl.drl_trainer import train_on_fix_interval_periods


def train_round_iter_func(round_n_per_target_day, target_avg_holding_days=[4, 6, 8]):
    # 作为训练起始随机动作时，平均换仓天数，该参数可能导致训练后的模型调仓频率变化
    round_n = 1
    for round_n_sub in range(round_n_per_target_day):
        for days in target_avg_holding_days:
            env_kwargs = dict(state_with_flag=True, fee_rate=0.001)
            agent_kwargs = dict(keep_last_action=math.pow(0.5, 1 / days), batch_size=512,
                                epsilon_memory_size=10, random_drop_best_cache_rate=0.1)
            num_episodes = 1000 + 200 * round_n_sub
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
    from ibats_common.example.data import load_data, OHLCAV_COL_NAME_LIST
    import functools
    from ibats_common.backend.factor import get_factor
    from ibats_common.example import get_trade_date_series, get_delivery_date_series
    instrument_type = 'RB'
    trade_date_series = get_trade_date_series(DATA_FOLDER_PATH)
    delivery_date_series = get_delivery_date_series(instrument_type, DATA_FOLDER_PATH)
    get_factor_func = functools.partial(get_factor,
                                        trade_date_series=trade_date_series, delivery_date_series=delivery_date_series)

    train_on_fix_interval_periods(
        md_loader_func=lambda range_to=None: load_data(
            'RB.csv', folder_path=DATA_FOLDER_PATH, index_col='trade_date', range_to=range_to)[OHLCAV_COL_NAME_LIST],
        get_factor_func=get_factor_func,
        train_round_kwargs_iter_func=functools.partial(train_round_iter_func, round_n_per_target_day=2), n_step=60,
        date_train_from='2015-09-30', offset='4M',
        max_process_count=2
    )
    # _test_train_round_iter_func(round_n_per_target_day=2)
