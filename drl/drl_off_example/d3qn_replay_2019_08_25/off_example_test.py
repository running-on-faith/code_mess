#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 19-9-5 下午2:43
@File    : off_example_test.py
@contact : mmmaaaggg@163.com
@desc    : 样本外测试
"""
# if True:
#     from ibats_common.backend.rl.utils import use_cup_only
#
#     use_cup_only()
from drl.d3qn_replay_2019_08_25.agent.main import MODEL_NAME, get_agent
from drl.validator import validate_bunch

if __name__ == "__main__":
    validate_bunch(
        model_name=MODEL_NAME, get_agent_func=get_agent,
        model_folder=r'/home/mg/github/code_mess/drl/drl_off_example/d3qn_replay_2019_08_25/output/2013-05-13/model',
        # model_folder=r'/home/mg/github/code_mess/drl/d3qn_replay_2019_08_25/agent/model',
        target_round_n=2,
        in_sample_date_line='2013-05-13',
        reward_2_csv=True,
    )
