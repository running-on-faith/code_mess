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
import numpy as np
from drl import DATA_FOLDER_PATH
from drl.d3qnr20200101.agent.main import MODEL_NAME, get_agent
from drl.drl_trainer import train_on_fix_interval_periods


def train_round_iter_func(round_n_per_target_day=2, target_avg_holding_days=[3, 5, 7]):
    """
    迭代器函数：每轮训练迭代参数 迭代器
    :param round_n_per_target_day:
    :param target_avg_holding_days:
    :return:
    """
    # 作为训练起始随机动作时，平均换仓天数，该参数可能导致训练后的模型调仓频率变化
    round_n = 1
    for round_n_sub in range(round_n_per_target_day):
        for days in target_avg_holding_days:
            env_kwargs = dict(state_with_flag=True, fee_rate=0.001)
            agent_kwargs = dict(
                target_avg_holding_days=days, batch_size=128, epochs=5, learning_rate=0.0001,
                epsilon_memory_size=10, random_drop_cache_rate=None,
                sin_step=np.pi/50, epsilon_decay=0.993, epsilon_min=0.01, epsilon_sin_max=0.1,
                build_model_layer_count=None, train_net_period=10, keep_epsilon_init_4_first_n=50,
            )
            num_episodes = 3000 + 200 * round_n_sub
            train_kwargs = dict(round_n=round_n, num_episodes=num_episodes, n_episode_pre_record=num_episodes // 8,
                                model_name=MODEL_NAME, get_agent_func=get_agent, output_reward_csv=False,
                                available_check_after_n_episode=2000)
            yield round_n, env_kwargs, agent_kwargs, train_kwargs
            round_n += 1


def _test_train_round_iter_func(round_n_per_target_day=3):
    logger = logging.getLogger(__name__)
    for round_n, env_kwargs, agent_kwargs, train_kwargs in train_round_iter_func(round_n_per_target_day):
        logger.error("round_n=%d, env_kwargs=%s, agent_kwargs%s, train_kwargs%s",
                     round_n, env_kwargs, agent_kwargs, train_kwargs)


def _test_train_on_each_period():
    from ibats_common.example.data import load_data, OHLCAV_COL_NAME_LIST
    import functools
    from ibats_common.backend.factor import get_factor
    from ibats_common.example import get_trade_date_series, get_delivery_date_series
    instrument_type = 'RB'
    trade_date_series = get_trade_date_series(DATA_FOLDER_PATH)
    delivery_date_series = get_delivery_date_series(instrument_type, DATA_FOLDER_PATH)
    get_factor_func = functools.partial(
        get_factor, trade_date_series=trade_date_series, delivery_date_series=delivery_date_series)

    train_on_fix_interval_periods(
        md_loader_func=lambda range_to=None: load_data(
            'RB.csv', folder_path=DATA_FOLDER_PATH, index_col='trade_date', range_to=range_to)[OHLCAV_COL_NAME_LIST],
        get_factor_func=get_factor_func,
        train_round_kwargs_iter_func=functools.partial(train_round_iter_func, round_n_per_target_day=4), n_step=60,
        date_train_from='2017-01-01', offset='4M',
        use_pool=True, max_process_count=4,
        date_period_count=1,  # None 如果需要训练全部日期
    )


if __name__ == '__main__':
    pass
    _test_train_on_each_period()
    # _test_train_round_iter_func(round_n_per_target_day=2)
