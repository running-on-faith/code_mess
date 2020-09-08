#! /usr/bin/env python3
"""
@author  : MG
@Time    : 2020/9/5 下午3:12
@File    : optimizer.py
@contact : mmmaaaggg@163.com
@desc    : 参数优化器
"""
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


def load_md_matlab(file_path) -> pd.DataFrame:
    df = pd.read_excel(file_path, header=None)
    dt_base = datetime.strptime('1899-12-30', '%Y-%m-%d')
    dt_s = df[0].apply(lambda x: dt_base + timedelta(days=x))
    df.index = dt_s
    df.rename(columns={1: 'open', 2: 'high', 3: 'low', 4: 'close'}, inplace=True)
    df = df[['open', 'high', 'low', 'close']]
    return df


def bulk_backtest_show_result(auto_open_html=True):
    """
    以带日期范围的上下界买卖策略为例测试程序是否可以正常运行
    """
    import itertools
    from params_optimizer.strategy import DoubleThresholdWithinPeriodsBSStrategy

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
    output_path = os.path.join('html', 'data.js')
    # md_df = load_md_matlab(file_path)
    # md_df = load_md(instrument_type='RB')
    contract_month = 1
    periods = generate_available_period(contract_month, date_from, date_to)
    params_kwargs_iter = [{
        "md_loader_kwargs": {"period": _[3]},
        "strategy_kwargs": {"buy_line": -5, "sell_line": 5, 'periods': periods, "lower_buy_higher_sell": False},
        "factor_kwargs": {"short": _[0], "long": _[1], "signal": _[2]}
    } for _ in itertools.product(
        range(6, 13, 2), range(14, 30, 3), range(5, 10),  # short, long, signal
        [5, 10, 15, 20, 30, 60, 120],  # file_path  BarSize=*
    )]
    result_dic = DoubleThresholdWithinPeriodsBSStrategy.bulk_run_on_range(
        lambda period: load_md_matlab(f'd:\\github\\matlab_mass\\data\\历年RB01BarSize={period}高开低收.xls'),
        factor_generator=factor_generator,
        params_kwargs_iter=params_kwargs_iter,
        date_from=date_from, date_to=date_to)

    data_2_js = []
    for n, (key, result_dic) in enumerate(result_dic.items(), start=1):
        logger.info("key: %s", key)
        dic = json.loads(key)
        data_dic = dic['factor_kwargs']
        data_dic.update(dic['md_loader_kwargs'])
        data_dic['calmar'] = result_dic['nav_stats'].calmar
        data_dic['cagr'] = result_dic['nav_stats'].cagr
        data_dic['daily_sharpe'] = result_dic['nav_stats'].daily_sharpe

        data_2_js.append(data_dic)
        # for n, (name, item) in enumerate(result_dic.items(), start=1):
        #     logger.debug("%3d) name:%s\n%s", n, name, item)
        logger.debug("%3d) %s", n, data_dic)

        reward_df = result_dic['reward_df']
        factor_kwargs = result_dic["params_kwargs"]["factor_kwargs"]
        md_loader_kwargs = result_dic["params_kwargs"]["md_loader_kwargs"]
        plot_twin(
            reward_df[["value", "value_fee0"]], reward_df[['close']],
            enable_show_plot=False,
            name=f"period{md_loader_kwargs['period']}"
                 f"_long{factor_kwargs['long']}"
                 f"_short{factor_kwargs['short']}"
                 f"_signal{factor_kwargs['signal']}"
        )

    # 保持测试结果数据
    with open(output_path, 'w') as f:
        f.write("var data = \n")
        json.dump([[
            _['short'], _['long'], _['signal'],
            _['calmar'], _['cagr'], _['daily_sharpe'], _['period']
        ] for _ in data_2_js], f)

    # 打开浏览器展示结果
    if auto_open_html:
        html_file_path = os.path.abspath('html/index.html')
        open_file_with_system_app(html_file_path)


if __name__ == "__main__":
    # _test_optimizer()
    # _test_generate_available_period()
    bulk_backtest_show_result()
