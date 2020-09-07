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
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import json
import logging
# noinspection PyUnresolvedReferences
import ffn  # NOQA
from ibats_common.analysis.plot import plot_twin
from ibats_common.backend.rl.emulator.market2 import ACTION_SHORT, ACTION_LONG, ACTION_CLOSE, ACTION_KEEP
from common.env import load_md

logger = logging.getLogger()


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
                          params_kwargs_iter: typing.Iterable[dict], date_from=None, date_to=None,
                          risk_free=0.03):
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
            reward_df = run_on_range(strategy, md_df, factors, date_from, date_to)
            nav_stats = reward_df['nav'].calc_stats()
            nav_stats.set_riskfree_rate(risk_free)
            nav_fee0_stats = reward_df['nav_fee0'].calc_stats()
            nav_fee0_stats.set_riskfree_rate(risk_free)
            result_dic[key] = {
                "reward_df": reward_df,
                "nav_stats": nav_stats,
                "nav_fee0_stats": nav_fee0_stats,
                "params_kwargs": params_kwargs,
            }

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
    """
    上下界买卖策略
    """

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


class DoubleThresholdWithinPeriodsBSStrategy(DoubleThresholdBSStrategy):
    """
    带有效日期范围的上下界买卖策略
    """

    def __init__(self, buy_line, sell_line, periods, lower_buy_higher_sell=True,
                 compare_index=1, mid_zone_action=ACTION_KEEP, date_index=0):
        super(DoubleThresholdWithinPeriodsBSStrategy, self).__init__(
            buy_line, sell_line, lower_buy_higher_sell,
            compare_index, mid_zone_action
        )
        self.periods = np.array(periods, dtype='M')
        assert self.periods.shape[1] == 2, 'periods 日期区间必须是一对日期'
        self.date_index = date_index
        self.periods_available = ~np.isnat(self.periods)

    def run(self, factors: np.ndarray) -> int:
        # 当前 bar 日期
        cur_date = factors[self.date_index]
        # 判断日期是否在有效日期区间范围内，
        # 如果不在，直接平仓，
        # 如果在，按照上下界进行判断
        is_ok = False
        for (date_from, date_to), (date_from_ok, date_to_ok) in zip(self.periods, self.periods_available):
            is_ok_from = (not date_from_ok) or date_from <= cur_date
            is_ok_to = (not date_to_ok) or cur_date <= date_to
            if is_ok_from and is_ok_to:
                is_ok = True
                break

        if not is_ok:
            action = ACTION_CLOSE
        else:
            action = super().run(factors)

        return action


def _test_optimizer():
    """
    以上下界买卖策略为例测试程序是否可以正常运行
    """

    def factor_generator(df: pd.DataFrame, short=12, long=26, signal=9):
        import talib
        close_s = df['close']
        dates = df.index.to_numpy()
        dif, dea, macd = talib.MACD(close_s, fastperiod=short, slowperiod=long, signalperiod=signal)
        factors = np.expand_dims(macd, axis=1)
        return factors

    date_from, date_to = '2013-01-01', '2018-12-31'
    md_df = load_md(instrument_type='RB')
    result_dic = DoubleThresholdBSStrategy.bulk_run_on_range(
        md_df,
        factor_generator=factor_generator,
        params_kwargs_iter=[{
            "strategy_kwargs": {"buy_line": 50, "sell_line": -50},
            "factor_kwargs": {"short": 12, "long": 26, "signal": 9}
        }],
        date_from=date_from, date_to=date_to)

    data_4_shown = []
    for key, result_dic in result_dic.items():
        print('\n', key, ':\n')
        dic = json.loads(key)
        data_dic = dic['factor_kwargs']
        data_dic['calmar'] = result_dic['nav_stats'].calmar
        data_4_shown.append(data_dic)
        for name, item in result_dic.items():
            print('\t', name)
            print('\t', item)

    print("data_4_shown:")
    for _ in data_4_shown:
        print(_)


def generate_available_period(contract_month: int, date_from_str: str, date_to_str: str) -> list:
    """
    生成合约对应的有效日期范围，与给点日期范围的交集
    该功能仅用于策略回测是对1月份连续合约等连续合约数据是使用
    根据合约生成日期范围规则，例如：
    1月合约，上一年8月1 ~ 11月31日
    5月合约，上一年12月1 ~ 3月31日
    9月合约，4月1日 ~ 7月31日
    """
    date_from = pd.to_datetime(date_from_str if date_from_str is not None else '2000-01-01')
    date_to = pd.to_datetime(date_to_str if date_from_str is not None else '2030-01-01')
    periods = []
    for range_year in range(date_from.year, date_to.year + 2):
        year, month = (range_year, contract_month - 5) if contract_month > 5 else (
            range_year - 1, contract_month + 12 - 5)
        range_from = pd.to_datetime(f"{year:4d}-{month:02d}-01")
        year, month = (range_year, contract_month - 1) if contract_month > 1 else (
            range_year - 1, 12)
        range_to = pd.to_datetime(f"{year:4d}-{month:02d}-01") - pd.to_timedelta(1, unit='D')
        # 与 date_from date_to 取交集
        if range_to < date_from or date_to < range_from:
            continue
        range_from = date_from if range_from < date_from < range_to else range_from
        range_to = date_to if range_from < date_to < range_to else range_to
        periods.append([str(range_from.date()), str(range_to.date())])

    return periods


def _test_generate_available_period():
    date_from, date_to = '2013-01-01', '2018-10-31'
    contract_month = 1
    periods = generate_available_period(contract_month, date_from, date_to)
    print(periods)


def load_md_matlab(file_path) -> pd.DataFrame:
    df = pd.read_excel(file_path, header=None)
    dt_base = datetime.strptime('1899-12-30', '%Y-%m-%d')
    dt_s = df[0].apply(lambda x: dt_base + timedelta(days=x))
    df.index = dt_s
    df.rename(columns={1: 'open', 2: 'high', 3: 'low', 4: 'close'}, inplace=True)
    df = df[['open', 'high', 'low', 'close']]
    return df


def bulk_backtest_show_result():
    """
    以带日期范围的上下界买卖策略为例测试程序是否可以正常运行
    """
    import itertools

    def factor_generator(df: pd.DataFrame, short=12, long=26, signal=9):
        import talib
        close_s = df['close']
        dates = df.index.to_numpy()
        dif, dea, macd = talib.MACD(close_s, fastperiod=short, slowperiod=long, signalperiod=signal)
        # factors = np.expand_dims(macd, axis=1)
        factors = np.array([_ for _ in zip(dates, macd)])
        return factors

    date_from, date_to = '2013-01-01', '2018-12-31'
    file_path = r'd:\github\matlab_mass\data\历年RB01BarSize=10高开低收.xls'
    output_path = r'c:\Users\zerenhe-lqb\Downloads\数据展示\数据展示\data2.json'
    md_df = load_md_matlab(file_path)
    # md_df = load_md(instrument_type='RB')
    contract_month = 1
    periods = generate_available_period(contract_month, date_from, date_to)
    params_kwargs_iter = [{
        "strategy_kwargs": {"buy_line": -5, "sell_line": 5, 'periods': periods, "lower_buy_higher_sell": False},
        "factor_kwargs": {"short": _[0], "long": _[1], "signal": _[2]}
    } for _ in itertools.product(range(8, 15), range(16, 30, 2), range(6, 13))]
    result_dic = DoubleThresholdWithinPeriodsBSStrategy.bulk_run_on_range(
        md_df,
        factor_generator=factor_generator,
        params_kwargs_iter=params_kwargs_iter,
        date_from=date_from, date_to=date_to)

    data_4_shown = []
    for n, (key, result_dic) in enumerate(result_dic.items(), start=1):
        logger.info("key: %s", key)
        dic = json.loads(key)
        data_dic = dic['factor_kwargs']
        data_dic['calmar'] = result_dic['nav_stats'].calmar
        data_4_shown.append(data_dic)
        # for n, (name, item) in enumerate(result_dic.items(), start=1):
        #     logger.debug("%3d) name:%s\n%s", n, name, item)
        logger.debug("%3d) %s", n, data_dic)

        reward_df = result_dic['reward_df']
        factor_kwargs = result_dic["params_kwargs"]["factor_kwargs"]
        plot_twin(
            reward_df[["value", "value_fee0"]], reward_df[['close']],
            enable_show_plot=False,
            name=f"long{factor_kwargs['long']}"
                 f"_short{factor_kwargs['short']}"
                 f"_signal{factor_kwargs['signal']}"
        )

    # 保持测试结果数据
    with open(output_path, 'w') as f:
        json.dump([[_['short'], _['long'], _['signal'], _['calmar']] for _ in data_4_shown], f)

    # 打开浏览器展示结果


if __name__ == "__main__":
    # _test_optimizer()
    # _test_generate_available_period()
    bulk_backtest_show_result()
