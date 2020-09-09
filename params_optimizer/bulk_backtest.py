#! /usr/bin/env python3
"""
@author  : MG
@Time    : 2020/9/5 下午3:12
@File    : optimizer.py
@contact : mmmaaaggg@163.com
@desc    : 参数优化器
"""
import functools
import itertools
import os
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import json
import logging
# noinspection PyUnresolvedReferences
import ffn  # NOQA
from ibats_common.analysis.plot import plot_twin
from ibats_utils.mess import open_file_with_system_app

from params_optimizer.html_generator import generate_html

logger = logging.getLogger()


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


@functools.lru_cache()
def load_md_matlab(file_path) -> pd.DataFrame:
    """
    加载 matlab 导出的 wind 的数据，日期轴做相应处理
    """
    logger.info("Load %s", file_path)
    df = pd.read_excel(file_path, header=None)
    dt_base = datetime.strptime('1899-12-30', '%Y-%m-%d')
    dt_s = df[0].apply(lambda x: dt_base + timedelta(days=x))
    df.index = dt_s
    df.rename(columns={1: 'open', 2: 'high', 3: 'low', 4: 'close'}, inplace=True)
    df = df[['open', 'high', 'low', 'close']]
    return df


def bulk_backtest_show_result(
        strategy_cls, date_from, date_to,
        contract_month, params_kwargs_iter, factor_generator,
        name='test', xls_data_dir_path=r"d:\github\matlab_mass\data",
        output_labels=['short', 'long', 'signal', 'calmar', 'cagr', 'daily_sharpe', 'period'],
        auto_open_html=True, tiny_random_shift_xyz=True):
    """
    以带日期范围的上下界买卖策略为例测试程序是否可以正常运行
    :strategy_cls 策略类
    :date_from 起始日期
    :date_to 截止日期
    :contract_month 合约月份
    :params_kwargs_iter 参数迭代器
    :factor_generator 因子生成器
    :name 名称，用于生产 html以及data文件时命名
    :xls_data_dir_path xls 文件目录
    :output_labels 输出标签
    :auto_open_html 自动打开 html
    :tiny_random_shift_xyz xyz三轴扰动
    """
    import itertools
    # 参数配置

    js_file_name = f'data_{name}.js'
    output_js_path = os.path.join('html', js_file_name)
    output_csv_path = os.path.join('html', f'data_{name}.csv')

    # file_path = r'd:\github\matlab_mass\data\历年RB01BarSize=10高开低收.xls'
    # md_df = load_md_matlab(file_path)
    # md_df = load_md(instrument_type='RB')
    # 数据文件所在目录
    # contract_month = 1  # 合约月份
    # 由于xyz都是整数，大数据情况下，很多点将会重叠在一个圆心位置，
    # 因此围绕圆心0.5为半径进行扰动
    # tiny_random_shift_xyz = True
    # 生成有效的时间段范围

    # 披露策略执行参数

    # 因子生成器

    # 策略批量运行
    result_dic = strategy_cls.bulk_run_on_range(
        lambda period: load_md_matlab(
            os.path.join(xls_data_dir_path, f'历年RB{contract_month:02d}BarSize={period}高开低收.xls')),
        factor_generator=factor_generator,
        params_kwargs_iter=params_kwargs_iter,
        date_from=date_from, date_to=date_to)

    # 策略结果整理
    data_2_js, data_len = [], len(result_dic)
    for n, (key, result_dic) in enumerate(result_dic.items(), start=1):
        logger.info("key: %s", key)
        dic = json.loads(key)
        data_dic = dic['factor_kwargs']
        if tiny_random_shift_xyz:
            # 由于xyz都是整数，大数据情况下，很多点将会重叠在一个圆心位置，
            # 因此围绕圆心0.5为半径进行扰动
            for _ in data_dic.keys():
                try:
                    data_dic[_] = data_dic[_] + np.random.random() - 0.5
                except TypeError:
                    pass

        data_dic.update(dic['md_loader_kwargs'])
        data_dic['calmar'] = result_dic['nav_stats'].calmar
        data_dic['cagr'] = result_dic['nav_stats'].cagr
        data_dic['daily_sharpe'] = result_dic['nav_stats'].daily_sharpe

        data_2_js.append(data_dic)
        # for n, (name, item) in enumerate(result_dic.items(), start=1):
        #     logger.debug("%3d) name:%s\n%s", n, name, item)
        logger.debug("%3d/%3d) %s", n, data_len, data_dic)

        reward_df = result_dic['reward_df']
        factor_kwargs = result_dic["params_kwargs"]["factor_kwargs"]
        md_loader_kwargs = result_dic["params_kwargs"]["md_loader_kwargs"]
        # 生成plot图
        plot_twin(
            reward_df[["value", "value_fee0"]], reward_df[['close']],
            enable_show_plot=False,
            name=f"period{md_loader_kwargs['period']}"
                 f"_long{factor_kwargs['long']}"
                 f"_short{factor_kwargs['short']}"
                 f"_signal{factor_kwargs['signal']}"
        )

    # 保持测试结果数据
    with open(output_js_path, 'w') as f:
        f.write("var data = \n")
        json.dump([[_[name] for name in output_labels] for _ in data_2_js], f)

    pd.DataFrame(data_2_js).to_csv(output_csv_path, index=None)

    # 输出 html
    html_file_path = generate_html(
        os.path.join('html', 'index.html'),
        labels=output_labels,
        x_label=output_labels[0],
        y_label=output_labels[6],
        z_label=output_labels[2],
        color_label=output_labels[3],
        symbol_size_label=output_labels[5],
        js_file_name=js_file_name,
    )

    # 打开浏览器展示结果
    if auto_open_html:
        html_file_path = os.path.abspath(html_file_path)
        open_file_with_system_app(html_file_path)


def do_macd_test():
    from params_optimizer.strategy import DoubleThresholdWithinPeriodsBSStrategy
    contract_month = 1
    date_from, date_to = '2013-01-01', '2020-12-31'
    periods = generate_available_period(contract_month, date_from, date_to)
    params_kwargs_iter = [{
        "md_loader_kwargs": {"period": _[3]},
        "strategy_kwargs": {"buy_line": -5, "sell_line": 5, 'periods': periods, "lower_buy_higher_sell": False},
        "factor_kwargs": {"short": _[0], "long": _[1], "signal": _[2]}
    } for _ in itertools.product(
        range(4, 13, 1), range(14, 30, 3), range(5, 10),  # short, long, signal
        [5, 10, 15, 20, 30, 60, 120],  # file_path  BarSize=*
    )]

    def factor_generator(df: pd.DataFrame, short=12, long=26, signal=9):
        import talib
        close_s = df['close']
        dates = df.index.to_numpy()
        dif, dea, macd = talib.MACD(close_s, fastperiod=short, slowperiod=long, signalperiod=signal)
        # factors = np.expand_dims(macd, axis=1)
        factors = np.array([_ for _ in zip(dates, macd)])
        return factors

    bulk_backtest_show_result(
        strategy_cls=DoubleThresholdWithinPeriodsBSStrategy,
        date_from=date_from,
        date_to=date_to,
        contract_month=contract_month,
        params_kwargs_iter=params_kwargs_iter,
        factor_generator=factor_generator,
        name='macd',
    )


def do_kdj_test():
    from params_optimizer.strategy import DoubleThresholdWithinPeriodsBSStrategy
    contract_month = 1
    date_from, date_to = '2013-01-01', '2020-12-31'
    periods = generate_available_period(contract_month, date_from, date_to)
    params_kwargs_iter = [{
        "md_loader_kwargs": {"period": _[3]},
        "strategy_kwargs": {"buy_line": 20, "sell_line": 80, 'periods': periods, "lower_buy_higher_sell": False},
        "factor_kwargs": {"fastk_period": _[0], "slowk_period": _[1], "slowd_period": _[2]}
    } for _ in itertools.product(
        np.arange(6.0, 15.0), np.arange(2.0, 6.0), np.arange(2.0, 6.0),  # fastk_period, slowk_period, slowd_period
        [5, 10, 15, 20, 30, 60, 120],  # file_path  BarSize=*
    )]

    def factor_generator(
            df: pd.DataFrame,
            fastk_period=9.0,
            slowk_period=3.0,
            slowd_period=3.0):
        import talib
        close_s = df['close'].to_numpy(dtype='f8')
        high_s = df['high'].to_numpy(dtype='f8')
        low_s = df['low'].to_numpy(dtype='f8')
        dates = df.index.to_numpy()
        # KDJ 值对应的函数是 STOCH
        slowk, slowd = talib.STOCH(
            close_s, high_s, low_s,
            fastk_period=fastk_period,
            slowk_period=slowk_period,
            slowk_matype=0.0,
            slowd_period=slowd_period,
            slowd_matype=0.0)
        # 求出J值，J = (3*K)-(2*D)
        slowj = list(map(lambda x, y: 3 * x - 2 * y, slowk, slowd))
        factors = np.array([_ for _ in zip(dates, slowk, slowd, slowj)])
        return factors

    bulk_backtest_show_result(
        strategy_cls=DoubleThresholdWithinPeriodsBSStrategy,
        date_from=date_from,
        date_to=date_to,
        contract_month=contract_month,
        params_kwargs_iter=params_kwargs_iter,
        factor_generator=factor_generator,
        name='kdj',
    )


if __name__ == "__main__":
    # _test_optimizer()
    # _test_generate_available_period()
    # bulk_backtest_show_result()
    # do_macd_test()
    do_kdj_test()
