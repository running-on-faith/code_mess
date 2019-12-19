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

from drl import DATA_FOLDER_PATH
from drl.d3qnr20191127.agent.main import MODEL_NAME, get_agent
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
            # 根据等比数列求和公式 Sn = a*(1-q^n)/(1-q), 当 Sn = 0.5, q= 0.5 时
            # a = Sn * (1 - q) / (1 - q^n) = 0.25 / (1 - 0.5^n)
            env_kwargs = dict(state_with_flag=True, fee_rate=0.001)
            agent_kwargs = dict(
                keep_last_action_rate=0.25 / (1 - 0.5 ** days), batch_size=128,
                epsilon_memory_size=10, random_drop_best_cache_rate=0.01,
                sin_step=0.2, epsilon_decay=0.993, epsilon_min=0.05, epsilon_sin_max=0.1
            )
            num_episodes = 3000 + 200 * round_n_sub
            train_kwargs = dict(round_n=round_n, num_episodes=num_episodes, n_episode_pre_record=num_episodes // 8,
                                model_name=MODEL_NAME, get_agent_func=get_agent, output_reward_csv=True)
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
        date_train_from='2017-01-01', offset='3M',
        use_pool=True, max_process_count=2,
        date_period_count=1,  # None 如果需要训练全部日期
    )


if __name__ == '__main__':
    pass
    _test_train_on_each_period()
    # _test_train_round_iter_func(round_n_per_target_day=2)
