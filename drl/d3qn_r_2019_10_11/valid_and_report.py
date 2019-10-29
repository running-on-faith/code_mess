#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 19-9-5 下午2:43
@File    : off_example_test_report.py
@contact : mmmaaaggg@163.com
@desc    : 样本外测试
"""
# if True:
#     from ibats_common.backend.rl.utils import use_cup_only
#
#     use_cup_only()
import functools

from ibats_common.backend.factor import get_factor
from ibats_common.example import get_trade_date_series, get_delivery_date_series
from ibats_common.example.data import OHLCAV_COL_NAME_LIST, load_data
from ibats_utils.mess import open_file_with_system_app

from drl import DATA_FOLDER_PATH
from drl.d3qn_r_2019_10_11.agent.main import MODEL_NAME, get_agent
from drl.validator import validate_bunch, auto_valid_and_report


def md_loader_func(range_to=None):
    return load_data(
        'RB.csv', folder_path=DATA_FOLDER_PATH, index_col='trade_date', range_to=range_to)[OHLCAV_COL_NAME_LIST]


instrument_type = 'RB'
trade_date_series = get_trade_date_series(DATA_FOLDER_PATH)
delivery_date_series = get_delivery_date_series(instrument_type, DATA_FOLDER_PATH)
get_factor_func = functools.partial(get_factor,
                                    trade_date_series=trade_date_series, delivery_date_series=delivery_date_series)


def valid_models_and_summary_report(auto_open_file=True, auto_open_summary_file=True):
    """针对model目录进行验证"""
    round_results_dic, file_path = validate_bunch(
        md_loader_func=md_loader_func,
        get_factor_func=get_factor_func,
        model_name=MODEL_NAME,
        get_agent_func=get_agent,
        model_folder=r'/home/mg/github/code_mess/drl/d3qn_r_2019_10_11/output/2017-01-26/model',
        # model_folder=r'/home/mg/github/code_mess/drl/d3qn_replay_2019_08_25/agent/model',
        in_sample_date_line='2013-11-08',
        reward_2_csv=True,
        target_round_n_list=[1],
    )
    if auto_open_summary_file and file_path is not None:
        open_file_with_system_app(file_path)
    for _, result_dic in round_results_dic.items():
        file_path = result_dic['summary_file_path']
        if auto_open_file and file_path is not None:
            open_file_with_system_app(file_path)


def valid_whole_episodes_and_summary_report(auto_open_file=False, auto_open_summary_file=False):
    """针对 output 目录，进行全面验证"""
    from ibats_utils.mess import is_windows_os
    if is_windows_os():
        output_folder = r'D:\WSPych\code_mess\drl\drl_off_example\d3qn_replay_2019_08_25\output'
    else:
        output_folder = r'/home/mg/github/code_mess/drl/d3qn_r_2019_10_11/output'
    # valid_models_and_summary_report()
    auto_valid_and_report(
        output_folder,
        md_loader_func=md_loader_func,
        get_factor_func=get_factor_func,
        model_name=MODEL_NAME,
        get_agent_func=get_agent,
        auto_open_file=auto_open_file, auto_open_summary_file=auto_open_summary_file)


if __name__ == "__main__":
    valid_whole_episodes_and_summary_report(
        auto_open_file=False,
        auto_open_summary_file=False
    )
