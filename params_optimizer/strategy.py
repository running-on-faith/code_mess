"""
@author  : MG
@Time    : 2020/9/8 9:21
@File    : strategy.py
@contact : mmmaaaggg@163.com
@desc    : 用于策略优化器的策略基类及一些子类
"""
import abc
import typing
import json
import logging
import pandas as pd
import numpy as np
# noinspection PyUnresolvedReferences
import ffn  # NOQA
from ibats_common.backend.rl.emulator.market2 import ACTION_SHORT, ACTION_LONG, ACTION_CLOSE, ACTION_KEEP
from ibats_common.backend.rl.emulator.account import Account, VERSION_V2

logger = logging.getLogger()


class SimpleStrategy:
    """
    策略类基类,用于回测环境的模拟多参数情况下的策略运行情况并生成绩效分析数据
    """

    def run_on_range(self, md_df: pd.DataFrame, factors_arr: np.ndarray,
                     date_from=None, date_to=None):
        """
        指定价格时间序列，因子矩阵, 时间段, 执行模拟运行,生成绩效统计结果
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
            action = self.run(next_state)
            next_state, reward, done = env.step(action)

        reward_df = env.generate_reward_df()
        return reward_df

    @abc.abstractmethod
    def run(self, factors: np.ndarray) -> int:
        pass

    @classmethod
    def bulk_run_on_range(cls, md_loader: typing.Callable[[str], pd.DataFrame],
                          factor_generator: typing.Callable[[pd.DataFrame, typing.Any], np.ndarray],
                          params_kwargs_iter: typing.Iterable[dict], date_from=None, date_to=None,
                          risk_free=0.03):
        """
        优化器
        """
        result_dic = {}
        for params_kwargs in params_kwargs_iter:
            md_loader_kwargs = params_kwargs.setdefault('md_loader_kwargs', {})
            strategy_kwargs = params_kwargs.setdefault('strategy_kwargs', {})
            factor_kwargs = params_kwargs.setdefault('factor_kwargs', {})
            md_df = md_loader(**md_loader_kwargs)
            factors = factor_generator(md_df, **factor_kwargs)
            key = json.dumps(params_kwargs)
            strategy = cls(**strategy_kwargs)
            reward_df = strategy.run_on_range(md_df, factors, date_from, date_to)
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
    from common.env import load_md

    def factor_generator(df: pd.DataFrame, short=12, long=26, signal=9):
        import talib
        close_s = df['close']
        dates = df.index.to_numpy()
        dif, dea, macd = talib.MACD(close_s, fastperiod=short, slowperiod=long, signalperiod=signal)
        factors = np.expand_dims(macd, axis=1)
        return factors

    date_from, date_to = '2013-01-01', '2018-12-31'
    result_dic = DoubleThresholdBSStrategy.bulk_run_on_range(
        load_md,  # md_df = load_md(instrument_type='RB')
        factor_generator=factor_generator,
        params_kwargs_iter=[{
            "md_loader_kwargs": {"instrument_type": "RB"},
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


if __name__ == "__main__":
    _test_optimizer()
