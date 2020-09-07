#! /usr/bin/env python3
"""
@author  : MG
@Time    : 2020/9/5 下午2:00
@File    : run_on_range.py
@contact : mmmaaaggg@163.com
@desc    : 指定价格时间序列, 时间段, 运行策略,并模拟运行,生成绩效统计结果
"""
import pandas as pd
import numpy as np
import logging
from ibats_common.backend.rl.emulator.account import Account, VERSION_V2
from params_optimizer.optimizer import SimpleStrategy

logger = logging.getLogger()


def run_on_range(strategy: SimpleStrategy, md_df: pd.DataFrame, factors_arr: np.ndarray, date_from=None, date_to=None):
    """

    :strategy 策略信息
    :md_df 行情数据 时间为索引
    :factor_df 因子数据, None则不传送因子数据,非空情况下需要与md_df第一维长度一致
    :date_from 起始日期, None则从行情头部开始
    :date_to 截止日期, None 则运行至行情结束
    :return:
    """
    if factors_arr is not None:
        # 判断第一维度是否相同
        assert md_df.shape[0] == factors_arr.shape[0], \
            f"md_df.shape[0]=={md_df.shape[0]}, but factors.shape[0]={factors_arr.shape[0]}"

    date_from = pd.to_datetime(date_from)
    date_to = pd.to_datetime(date_to)
    matches_from = None if date_from is None else date_from <= md_df.index
    matches_to = None if date_to is None else md_df.index <= date_to
    if matches_from is not None and matches_to is not None:
        matches = matches_from & matches_to
    elif matches_from is None and matches_to is not None:
        matches = matches_to
    elif matches_from is not None and matches_to is None:
        matches = matches_from
    else:
        matches = None

    if matches is not None:
        sub_md_df = md_df[matches]
        sub_factors_arr = factors_arr[np.where(matches)[0]]
    else:
        sub_md_df = md_df
        sub_factors_arr = factors_arr

    logger.info("date_from=%s, date_to=%s, data_length=%d, factor_shape=%s",
                date_from, date_to, sub_md_df.shape[0], sub_factors_arr.shape)
    env = Account(sub_md_df, sub_factors_arr, expand_dims=False, state_with_flag=False, version=VERSION_V2)
    next_state = env.reset()
    done = False
    while not done:
        action = strategy.run(next_state)
        next_state, reward, done = env.step(action)

    reward_df = env.generate_reward_df()
    return reward_df


if __name__ == "__main__":
    pass
