#! /usr/bin/env python3
"""
@author  : MG
@Time    : 2020/9/5 下午3:12
@File    : optimizer.py
@contact : mmmaaaggg@163.com
@desc    : 参数优化器
"""
import abc
import typing
import pandas as pd
import numpy as np
import json
from common.env import load_md
from params_optimizer.run_on_range import run_on_range


class SimpleStrategy:
    """
    见到策略类,用于基于 run_on_range 回测环境的策略类
    """

    @abc.abstractmethod
    def run(self, factors: np.ndarray) -> int:
        pass

    @classmethod
    def bulk_run_on_range(cls, md_df: pd.DataFrame,
                          factor_generator: typing.Callable[[pd.DataFrame, typing.Optional], np.ndarray],
                          params_kwargs_iter: typing.Iterable[dict], date_from=None, date_to=None):
        """
        优化器
        """
        result_dic = {}
        for params_kwargs in params_kwargs_iter:
            strategy_kwargs = params_kwargs.setdefault('strategy_kwargs', {})
            factor_kwargs = params_kwargs.setdefault('factor_kwargs', {})
            factors = factor_generator(md_df, **factor_kwargs)
            key = json.loads(params_kwargs)
            strategy = cls(**strategy_kwargs)
            result = run_on_range(strategy, md_df, factors, date_from, date_to)
            result_dic[key] = result

        return result_dic


class ZeroLineBSStrategy(SimpleStrategy):
    def run(self, factors: np.ndarray) -> int:
        from ibats_common.backend.rl.emulator.market2 import ACTION_SHORT, ACTION_LONG, ACTION_CLOSE
        factor = factors[-1, 0]
        if factor > 0:
            action = ACTION_LONG
        elif factor < 0:
            action = ACTION_SHORT
        else:
            action = ACTION_CLOSE

        return action


def _test_optimizer():
    def factor_generator(md: pd.DataFrame, short, long, signal):
        return np.zeros(1)

    date_from, date_to = '2013-01-01', '2018-12-31'
    md_df = load_md(instrument_type='RB')
    result_dic = ZeroLineBSStrategy.bulk_run_on_range(
        md_df,
        factor_generator=factor_generator,
        params_kwargs_iter=[{"strategy_kwargs": {}, "factor_kwargs": {}}],
        date_from=date_from, date_to=date_to)
    return result_dic


if __name__ == "__main__":
    pass
