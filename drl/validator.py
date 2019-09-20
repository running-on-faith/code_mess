#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 19-9-3 下午5:07
@File    : validator.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
import logging
import os
from collections import defaultdict

import pandas as pd
from ibats_common.analysis.plot import plot_twin
from ibats_common.backend.factor import get_factor, transfer_2_batch
from ibats_common.backend.rl.emulator.account import Account
from ibats_common.example.data import load_data, OHLCAV_COL_NAME_LIST
from ibats_utils.mess import date_2_str, open_file_with_system_app, str_2_date

from analysis.summary import summary_analysis_result_dic_2_docx
from drl import DATA_FOLDER_PATH


def load_model_and_predict_through_all(md_df, batch_factors, model_name, get_agent_func,
                                       tail_n=1, show_plot=True, model_path="model/weights_1.h5", key=None):
    """加载 model_path 模型，对batch_factors计算买卖决策，对 md_df 行情进行模拟，回测检验"""
    logger = logging.getLogger(__name__)
    if tail_n is not None and tail_n > 0:
        states = batch_factors[-tail_n:]
        md_df = md_df.iloc[-tail_n:]
    else:
        states = batch_factors

    env = Account(md_df, data_factors=states, state_with_flag=True)
    agent = get_agent_func(input_shape=states.shape)
    max_date = max(md_df.index)
    max_date_str = date_2_str(max_date)
    agent.restore_model(path=model_path)
    # logger.debug("模型：%s 加载完成，样本内测试开始", model_path)

    # 单步执行
    done, state, num = False, env.reset(), 0
    while not done:
        num += 1
        action = agent.choose_action_deterministic(state)
        state, reward, done = env.step(action)
        if done:
            if num + 1 < md_df.shape[0]:
                logger.debug('执行循环 %d / %d 次', num, md_df.shape[0])
            break

    reward_df = env.plot_data()
    if show_plot:
        value_df = reward_df[['value', 'value_fee0']] / env.A.init_cash
        title = f'{model_name}_predict_{max_date_str}' if key is None else \
            f'{model_name}_predict_episode_{key}_{max_date_str}'
        plot_twin(value_df, md_df["close"], name=title, folder_path='images')

    action_count = reward_df['action_count'].iloc[-1]
    logger.debug("累计操作 %d 次, 平均持仓时间 %.2f 天, 净值：%.4f, 模型：%s，",
                 action_count, num / action_count * 2, reward_df['value'].iloc[-1] / env.A.init_cash, model_path, )
    return reward_df


def validate_bunch(md_loader, model_name, get_agent_func, in_sample_date_line, model_folder='model', read_csv=True,
                   reward_2_csv=False, target_round_n_list: (None, list) = None, n_step=60, in_sample_only=False,
                   **analysis_kwargs):
    """

    :param md_loader: 数据加载器
    :param model_name: 模型名称
    :param get_agent_func: drl agent 生成器
    :param in_sample_date_line: 样本内截止日期
    :param model_folder: 模型目录
    :param read_csv: 是否读取各 episode 相应 reward_df 的 .csv 文件
    :param reward_2_csv:  是否生产各 episode 相应 reward_df 的 .csv 文件
    :param target_round_n_list: 目标 round_n 列表，默认 None 代表全部
    :param n_step: factor 生成是的 step， 该step 需要与模型训练时 step 值保持一致，否则模型将无法运行
    :param in_sample_only: 是否只对样本内数据继续验证
    :param analysis_kwargs: reward 分析相关参数
    :return:
    """
    logger = logging.getLogger(__name__)
    analysis_kwargs['in_sample_date_line'] = in_sample_date_line
    analysis_kwargs['in_sample_only'] = in_sample_only
    # 建立相关数据
    # OHLCAV_COL_NAME_LIST = ["open", "high", "low", "close", "amount", "volume"]
    # md_df = load_data('RB.csv',
    #                   folder_path=DATA_FOLDER_PATH,
    #                   ).set_index('trade_date')[OHLCAV_COL_NAME_LIST]
    # 如果 in_sample_only 则只加载样本内行情数据
    md_df = md_loader(in_sample_date_line if in_sample_only else None)
    md_df.index = pd.DatetimeIndex(md_df.index)
    factors_df = get_factor(md_df, dropna=True)
    df_index, df_columns, batch_factors = transfer_2_batch(factors_df, n_step=n_step)
    data_factors, shape = batch_factors, batch_factors.shape
    logger.info('batch_factors.shape=%s', shape)
    max_date = max(md_df.index)
    max_date_str = date_2_str(max_date)
    # shape = [data_arr_batch.shape[0], 5, int(n_step / 5), data_arr_batch.shape[2]]
    # data_factors = np.transpose(data_arr_batch.reshape(shape), [0, 2, 3, 1])
    # print(data_arr_batch.shape, '->', shape, '->', data_factors.shape)
    md_df = md_df.loc[df_index, :]
    round_n_episode_model_path_dic = defaultdict(dict)
    for file_name in os.listdir(model_folder):
        file_path = os.path.join(model_folder, file_name)
        if os.path.isdir(file_path):
            continue
        file_name_no_extension, extension = os.path.splitext(file_name)
        if extension != '.h5':
            continue
        # _, file_name_no_extension = os.path.split(file_name)
        # model_path = f"model/weights_1.h5"
        _, round_n, episode = file_name_no_extension.split('_')
        round_n = int(round_n)
        if target_round_n_list is not None and len(target_round_n_list) > 0 and int(round_n) not in target_round_n_list:
            continue
        episode = int(episode)
        round_n_episode_model_path_dic[round_n][episode] = file_path

    if len(round_n_episode_model_path_dic) == 0:
        logger.info('target_round_n=%d 没有可加载的模型', target_round_n_list)
        return

    index_col = ['trade_date']
    round_n_list = list(round_n_episode_model_path_dic.keys())
    round_n_list.sort()
    round_results_dic = defaultdict(dict)
    for round_n in round_n_list:
        episode_reward_df_dic = {}
        episode_list = list(round_n_episode_model_path_dic[round_n].keys())
        episode_list.sort()
        episode_count = len(episode_list)
        for num, episode in enumerate(episode_list, start=1):
            file_path = str(round_n_episode_model_path_dic[round_n][episode])
            logger.debug('%2d/%2d ) %4d -> %s', num, episode_count, episode, file_path)
            reward_file_path = os.path.join(model_folder, f'reward_{round_n}_{episode}.csv')
            if read_csv and os.path.exists(reward_file_path):
                reward_df = pd.read_csv(reward_file_path, index_col=index_col, parse_dates=index_col)
                if reward_df.shape[0] == 0:
                    continue
            else:
                reward_df = load_model_and_predict_through_all(
                    md_df, data_factors, model_name, get_agent_func,
                    tail_n=0, model_path=file_path, key=episode, show_plot=False)
                if reward_df.shape[0] == 0:
                    continue
                if reward_2_csv:
                    reward_df.to_csv(reward_file_path)

            episode_reward_df_dic[episode] = reward_df

        # 创建 word 文档
        from analysis.summary import summary_rewards_2_docx
        from analysis.analysis import analysis_rewards_with_md
        # 模型相关参数
        param_dic = dict(model_name=model_name,
                         in_sample_date_line=date_2_str(in_sample_date_line),
                         round_n=round_n,
                         model_folder=model_folder,
                         n_step=n_step,
                         )
        title_header = f"{model_name}_{date_2_str(in_sample_date_line)}{'_i' if in_sample_only else ''}_{round_n}"
        analysis_kwargs['title_header'] = title_header
        analysis_kwargs['round_n'] = round_n
        analysis_kwargs['episode_model_path_dic'] = round_n_episode_model_path_dic[round_n]
        analysis_result_dic = analysis_rewards_with_md(
            episode_reward_df_dic, md_df, **analysis_kwargs)
        summary_file_path = summary_rewards_2_docx(param_dic, analysis_result_dic, title_header)

        logger.debug('文件路径[%d]：%s', round_n, summary_file_path)
        round_results_dic[round_n] = dict(
            param_dic=param_dic,
            analysis_kwargs=analysis_kwargs,
            title_header=title_header,
            summary_file_path=summary_file_path,
            analysis_result_dic=analysis_result_dic,
        )

    title_header = f"{model_name}_{date_2_str(in_sample_date_line)}{'_i' if in_sample_only else ''}"
    file_path = summary_analysis_result_dic_2_docx(round_results_dic, title_header)

    return round_results_dic, file_path


def _test_validate_bunch(auto_open_file=True):
    from drl.d3qn_replay_2019_08_25.agent.main import get_agent, MODEL_NAME
    round_results_dic, file_path = validate_bunch(
        md_loader=lambda range_to=None: load_data(
            'RB.csv', folder_path=DATA_FOLDER_PATH, index_col='trade_date', range_to=range_to)[OHLCAV_COL_NAME_LIST],
        model_name=MODEL_NAME, get_agent_func=get_agent,
        model_folder=r'/home/mg/github/code_mess/drl/drl_off_example/d3qn_replay_2019_08_25/output/2013-11-08/model',
        # model_folder=r'/home/mg/github/code_mess/drl/d3qn_replay_2019_08_25/agent/model',
        in_sample_date_line='2013-11-08',
        reward_2_csv=True,
        target_round_n_list=[1],
    )
    if auto_open_file and file_path is not None:
        open_file_with_system_app(file_path)

    for _, result_dic in round_results_dic.items():
        file_path = result_dic['summary_file_path']
        if auto_open_file and file_path is not None:
            open_file_with_system_app(file_path)


def auto_valid_and_report(output_folder, model_name, get_agent, auto_open_file=False, auto_open_summary_file=True):
    logger = logging.getLogger(__name__)
    date_model_folder_dic = {}
    for file_name in os.listdir(output_folder):
        folder_path = os.path.join(output_folder, file_name)
        if not os.path.isdir(folder_path):
            continue
        try:
            in_sample_date_line = str_2_date(file_name)
        except:
            logger.debug('跳过 %s 目录', folder_path)
            continue
        import datetime
        if not isinstance(in_sample_date_line, datetime.date):
            continue
        model_folder = os.path.join(folder_path, 'model')
        date_model_folder_dic[in_sample_date_line] = model_folder

    date_list = list(date_model_folder_dic.keys())
    date_list.sort()
    date_round_results_dic = {}
    for in_sample_date_line in date_list:
        model_folder = date_model_folder_dic[in_sample_date_line]
        round_results_dic, file_path = validate_bunch(
            md_loader=lambda range_to=None: load_data(
                'RB.csv', folder_path=DATA_FOLDER_PATH, index_col='trade_date', range_to=range_to)[
                OHLCAV_COL_NAME_LIST],
            model_name=model_name, get_agent_func=get_agent,
            model_folder=model_folder,
            in_sample_date_line=in_sample_date_line,
            reward_2_csv=True,
            show_plot_141=False
        )
        date_round_results_dic[in_sample_date_line] = round_results_dic
        if auto_open_summary_file and file_path is not None:
            open_file_with_system_app(file_path)
        for _, result_dic in round_results_dic.items():
            file_path = result_dic['summary_file_path']
            if auto_open_file and file_path is not None:
                open_file_with_system_app(file_path)

    # 将各个日期的有效 model 路径合并成一个 DataFrame
    # date  round   episode file_path
    df_dic_list, key = [], 'available_episode_model_path_dic'
    for in_sample_date_line, round_results_dic in date_round_results_dic.items():
        for round_n, result_dic in round_results_dic.items():
            if key in result_dic['analysis_result_dic']:
                for episode, model_path in result_dic['analysis_result_dic'][key].items():
                    df_dic_list.append(
                        dict(date=in_sample_date_line, round=round_n, episode=episode, file_path=model_path))

    df = pd.DataFrame(df_dic_list)[['date', 'round', 'episode', 'file_path']]
    file_name = f'available_model_path.csv'
    file_path = os.path.join(output_folder, file_name)
    df.to_csv(file_path, index=False)
    logger.info('有效模型路径输出到文件： %s', file_path)


def _test_auto_valid_and_report(output_folder, auto_open_file=True):
    from drl.d3qn_replay_2019_08_25.agent.main import get_agent, MODEL_NAME
    auto_valid_and_report(output_folder, MODEL_NAME, get_agent, auto_open_file=auto_open_file)


if __name__ == "__main__":
    _test_validate_bunch()
    # _test_auto_valid_and_report(
    #     output_folder='/home/mg/github/code_mess/drl/drl_off_example/d3qn_replay_2019_08_25/output',
    #     auto_open_file=False)
