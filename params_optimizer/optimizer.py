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
from ibats_common.backend.rl.emulator.market2 import ACTION_SHORT, ACTION_LONG, ACTION_CLOSE, ACTION_KEEP
from common.env import load_md


class SimpleStrategy:
    """
    见到策略类,用于基于 run_on_range 回测环境的策略类
    """

    @abc.abstractmethod
    def run(self, factors: np.ndarray) -> int:
        pass

    @classmethod
    def bulk_run_on_range(cls, md_df: pd.DataFrame,
                          factor_generator: typing.Callable[[pd.DataFrame, typing.Any], np.ndarray],
                          params_kwargs_iter: typing.Iterable[dict], date_from=None, date_to=None):
        """
        优化器
        """
        from params_optimizer.run_on_range import run_on_range
        result_dic = {}
        for params_kwargs in params_kwargs_iter:
            strategy_kwargs = params_kwargs.setdefault('strategy_kwargs', {})
            factor_kwargs = params_kwargs.setdefault('factor_kwargs', {})
            factors = factor_generator(md_df, **factor_kwargs)
            key = json.dumps(params_kwargs)
            strategy = cls(**strategy_kwargs)
            result = run_on_range(strategy, md_df, factors, date_from, date_to)
            result_dic[key] = result

        return result_dic


class ZeroLineBSStrategy(SimpleStrategy):
    def run(self, factors: np.ndarray) -> int:
        factor = factors[-1, 0]
        if factor > 0:
            action = ACTION_LONG
        elif factor < 0:
            action = ACTION_SHORT
        else:
            action = ACTION_CLOSE

        return action


class DoubleThresholdBSStrategy(SimpleStrategy):
    def __init__(self, buy_line, sell_line, lower_buy_higher_sell=True,
                 compare_index=0, mid_zone_action=ACTION_KEEP):
        """

        :buy_line 买入阈值
        :sell_line 卖出阈值
        :higher_buy_lower_sell True 低买高卖,False 相反
        :compare_index 用于比较的因子的索引位置
        :mid_zone_action 中间地带采取的动作
        :return:
        """
        self.buy_line = buy_line
        self.sell_line = sell_line
        self.lower_buy_higher_sell = lower_buy_higher_sell
        self.compare_index = compare_index
        self.mid_zone_action = mid_zone_action

    def run(self, factors: np.ndarray) -> int:
        factor = factors[self.compare_index]
        if self.lower_buy_higher_sell:
            if factor < self.buy_line:
                action = ACTION_LONG
            elif factor > self.sell_line:
                action = ACTION_SHORT
            else:
                action = self.mid_zone_action
        else:
            if factor > self.buy_line:
                action = ACTION_LONG
            elif factor < self.sell_line:
                action = ACTION_SHORT
            else:
                action = self.mid_zone_action

        return action


def _test_optimizer():
    def factor_generator(df: pd.DataFrame, short=12, long=26, signal=9):
        import talib
        close_s = df['close']
        dif, dea, macd = talib.MACD(close_s, fastperiod=12, slowperiod=24, signalperiod=9)
        factors = np.expand_dims(macd, axis=1)
        return factors

    date_from, date_to = '2013-01-01', '2018-12-31'
    md_df = load_md(instrument_type='RB')
    result_dic = DoubleThresholdBSStrategy.bulk_run_on_range(
        md_df,
        factor_generator=factor_generator,
        params_kwargs_iter=[{
            "strategy_kwargs": {"buy_line": 80, "sell_line": 20},
            "factor_kwargs": {}
        }],
        date_from=date_from, date_to=date_to)

    print(result_dic)


if __name__ == "__main__":
    _test_optimizer()
