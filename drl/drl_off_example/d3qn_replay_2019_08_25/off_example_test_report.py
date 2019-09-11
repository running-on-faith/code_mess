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
from ibats_utils.mess import open_file_with_system_app

from drl.d3qn_replay_2019_08_25.agent.main import MODEL_NAME, get_agent
from drl.validator import validate_bunch, auto_valid_and_report


def valid_model(auto_open_file=True):
    round_summary_file_path_dic = validate_bunch(
        model_name=MODEL_NAME, get_agent_func=get_agent,
        model_folder=r'/home/mg/github/code_mess/drl/drl_off_example/d3qn_replay_2019_08_25/output/2013-11-08/model',
        # model_folder=r'/home/mg/github/code_mess/drl/d3qn_replay_2019_08_25/agent/model',
        in_sample_date_line='2013-11-08',
        reward_2_csv=True,
        target_round_n_list=[1],
    )
    for _, file_path in round_summary_file_path_dic.items():
        if auto_open_file and file_path is not None:
            open_file_with_system_app(file_path)


def valid_whole(output_folder, auto_open_file=False):
    auto_valid_and_report(output_folder, MODEL_NAME, get_agent, auto_open_file=auto_open_file)


if __name__ == "__main__":
    # valid_model()
    valid_whole(
        output_folder='/home/mg/github/code_mess/drl/drl_off_example/d3qn_replay_2019_08_25/output',
        auto_open_file=False)
