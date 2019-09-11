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
from ibats_common.example.data import load_data
from ibats_utils.mess import date_2_str, open_file_with_system_app, str_2_date

from drl import DATA_FOLDER_PATH


def load_model_and_predict_through_all(md_df, batch_factors, model_name, get_agent_func,
                                       tail_n=1, show_plot=True, model_path="model/weights_1.h5", key=None):
    """加载模型，进行样本内训练"""
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


def validate_bunch(model_name, get_agent_func, model_folder='model', read_csv=True, reward_2_csv=False,
                   target_round_n_list: (None, list) = None, n_step=60, **analysis_kwargs):
    logger = logging.getLogger(__name__)
    # 建立相关数据
    ohlcav_col_name_list = ["open", "high", "low", "close", "amount", "volume"]

    md_df = load_data('RB.csv',
                      folder_path=DATA_FOLDER_PATH,
                      ).set_index('trade_date')[ohlcav_col_name_list]
    md_df.index = pd.DatetimeIndex(md_df.index)
    factors_df = get_factor(md_df, dropna=True)
    df_index, df_columns, data_arr_batch = transfer_2_batch(factors_df, n_step=n_step)
    data_factors, shape = data_arr_batch, data_arr_batch.shape
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

    episode_reward_df_dic = {}
    index_col = ['trade_date']
    round_n_list = list(round_n_episode_model_path_dic.keys())
    round_n_list.sort()
    round_summary_file_path_dic = {}
    for round_n in round_n_list:
        episode_list = list(round_n_episode_model_path_dic[round_n].keys())
        episode_list.sort()
        episode_count = len(episode_list)
        for num, episode in enumerate(episode_list, start=1):
            file_path = str(round_n_episode_model_path_dic[round_n][episode])
            logger.debug('%2d/%2d ) %4d -> %s', num, episode_count, episode, file_path)
            reward_file_path = os.path.join(model_folder, f'reward_{round_n}_{episode}.csv')
            if read_csv and os.path.exists(reward_file_path):
                reward_df = pd.read_csv(reward_file_path, index_col=index_col, parse_dates=index_col)
            else:
                reward_df = load_model_and_predict_through_all(
                    md_df, data_factors, model_name, get_agent_func,
                    tail_n=0, model_path=file_path, key=episode, show_plot=False)

            if reward_df.shape[0] == 0:
                continue
            if reward_2_csv:
                reward_df.to_csv(reward_file_path)

            episode_reward_df_dic[episode] = reward_df

        title_header = f"{model_name}_{max_date_str}_{round_n}"
        analysis_kwargs['title_header'] = title_header
        # analysis_rewards(episode_reward_df_dic, md_df, **analysis_kwargs)
        from analysis.summary import analysis_rewards_with_md, summary_4_rewards
        result_dic = analysis_rewards_with_md(
            episode_reward_df_dic, md_df, show_plot_141=True, **analysis_kwargs)
        doc_folder_path = os.path.join(model_folder, os.pardir, 'reports')
        os.makedirs(doc_folder_path, exist_ok=True)
        summary_file_path = summary_4_rewards(result_dic, md_df, title_header)
        round_summary_file_path_dic[round_n] = summary_file_path
        logger.debug('文件路径[%d]：%s', round_n, summary_file_path)
    return round_summary_file_path_dic


def analysis_rewards(episode_reward_df_dic: dict, md_df, title_header, in_sample_date_line=None,
                     show_plot_together=True):
    """分析 rewards，生成报告(以及被summary_4_rewards 取代)"""
    logger = logging.getLogger(__name__)
    episode_list = list(episode_reward_df_dic.keys())
    episode_list.sort()
    episode_count = len(episode_list)
    import matplotlib.pyplot as plt
    fig, ax = plt.figure(figsize=(10, 16)), None  #
    if show_plot_together:
        ax = fig.add_subplot(211)
        # 前两个以及最后一个输出，其他的能整除才输出
        mod = int(episode_count / 5)
        value_df = pd.DataFrame({f'{episode}_v': episode_reward_df_dic[episode]['value']
                                 for num, episode in enumerate(episode_list)
                                 if episode_reward_df_dic[episode].shape[0] > 0 and (
                                         num % mod == 0 or num in (1, episode_count - 1))})
        value_fee0_df = pd.DataFrame({f'{episode}_0': episode_reward_df_dic[episode]['value_fee0']
                                      for num, episode in enumerate(episode_list)
                                      if episode_reward_df_dic[episode].shape[0] > 0 and (
                                              num % mod == 0 or num in (1, episode_count - 1))})

        # 例如：d3qn_in_sample_205-01-01_r0_2019-09-10
        title = title_header + f"{'_' + date_2_str(in_sample_date_line) if in_sample_date_line is not None else ''}"
        plot_twin([value_df, value_fee0_df], md_df['close'], ax=ax, name=title,
                  enable_save_plot=False, enable_show_plot=False, do_clr=False,
                  in_sample_date_line=in_sample_date_line)

    # 展示训练曲线
    if len(episode_reward_df_dic) > 0:
        predict_result_dic = {_: val.iloc[-1, :] for _, val in episode_reward_df_dic.items()}
        if ax is not None:
            ax = fig.add_subplot(212)
        predict_result_df = pd.DataFrame(predict_result_dic).T.sort_index()
        title = f'{title_header}_summary'
        plot_twin(predict_result_df[['value', 'value_fee0']], predict_result_df['action_count'],
                  ax=ax, name=title, y_scales_log=[False, True], folder_path='images')
        logger.debug("predict_result_df=\n%s", predict_result_df)


def _test_validate_bunch(auto_open_file=True):
    from drl.d3qn_replay_2019_08_25.agent.main import get_agent, MODEL_NAME

    round_summary_file_path_dic = validate_bunch(
        model_name=MODEL_NAME, get_agent_func=get_agent,
        model_folder=r'/home/mg/github/code_mess/drl/drl_off_example/d3qn_replay_2019_08_25/output/2013-11-08/model',
        # model_folder=r'/home/mg/github/code_mess/drl/d3qn_replay_2019_08_25/agent/model',
        in_sample_date_line='2013-11-08',
        reward_2_csv=True,
    )
    for _, file_path in round_summary_file_path_dic.items():
        if auto_open_file and file_path is not None:
            open_file_with_system_app(file_path)


def auto_valid_and_report(output_folder, model_name, get_agent, auto_open_file=False):
    date_model_folder_dic = {}
    for file_name in os.listdir(output_folder):
        file_path = os.path.join(output_folder, file_name)
        if not os.path.isdir(file_path):
            continue
        in_sample_date_line = str_2_date(file_name)
        import datetime
        if not isinstance(in_sample_date_line, datetime.date):
            continue
        model_folder = os.path.join(file_path, 'model')
        date_model_folder_dic[in_sample_date_line] = model_folder

    date_list = list(date_model_folder_dic.keys())
    date_list.sort()
    for in_sample_date_line in date_list:
        model_folder = date_model_folder_dic[in_sample_date_line]
        round_summary_file_path_dic = validate_bunch(
            model_name=model_name, get_agent_func=get_agent,
            model_folder=model_folder,
            in_sample_date_line=in_sample_date_line,
            reward_2_csv=True,
        )
        for _, file_path in round_summary_file_path_dic.items():
            if auto_open_file and file_path is not None:
                open_file_with_system_app(file_path)


def _test_auto_valid_and_report(output_folder, auto_open_file=True):
    from drl.d3qn_replay_2019_08_25.agent.main import get_agent, MODEL_NAME
    auto_valid_and_report(output_folder, MODEL_NAME, get_agent, auto_open_file=auto_open_file)


if __name__ == "__main__":
    # _test_validate_bunch()
    _test_auto_valid_and_report(
        output_folder='/home/mg/github/code_mess/drl/drl_off_example/d3qn_replay_2019_08_25/output',
        auto_open_file=False)
