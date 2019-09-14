#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 19-9-11 下午5:22
@File    : analysis.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
import logging
import os

import ffn
import numpy as np
import pandas as pd
from ibats_common.analysis.plot import plot_twin
from ibats_utils.mess import open_file_with_system_app, date_2_str

from analysis.summary import summary_rewards_2_docx
from drl import DATA_FOLDER_PATH

logger = logging.getLogger(__name__)
logger.debug('import %s', ffn)


def analysis_rewards_with_md(episode_reward_df_dic, md_df, title_header, in_sample_date_line, enable_show_plot=False,
                             enable_save_plot=True, show_plot_141=False, risk_free=0.03, **kwargs):
    """分析 rewards 绩效"""
    # 整理参数
    # cache_folder_path = r'/home/mg/github/code_mess/output/cache'
    result_dic = {}
    day_span_list = [5, 10, 20, 60]
    close_s = md_df['close']
    episode_count, episode_list = len(episode_reward_df_dic), list(episode_reward_df_dic.keys())
    if episode_count == 0:
        return result_dic
    episode_list.sort()
    enable_kwargs = dict(enable_save_plot=enable_save_plot, enable_show_plot=enable_show_plot, figsize=(5.4, 6.8))
    in_sample_date_line = pd.to_datetime(in_sample_date_line)

    def calc_reward_nav_value(reward_df: pd.DataFrame, baseline=None):
        df = reward_df[['value', 'value_fee0', 'close', 'action_count']]
        df['avg_holding'] = df['action_count'] / df.shape[0]
        if baseline is not None:
            df = df[df.index <= baseline]
        ret_s = df.iloc[-1, :].copy()
        ret_s[['value', 'value_fee0', 'close']] /= df.iloc[0, :][['value', 'value_fee0', 'close']]
        return ret_s

    # 计算 样本内 以及 day_span_list 日 Episode 对应的 value
    # 建立 每日各个 episode value 值 DataFrame
    date_episode_value_df = pd.DataFrame(
        {episode: reward_df['value'] / reward_df['value'].iloc[0]
         for episode, reward_df in episode_reward_df_dic.items()}
    ).sort_index().sort_index(axis=1).merge(close_s, right_index=True, left_index=True)
    # 将 close 净值化
    date_episode_value_df['close'] /= date_episode_value_df['close'].iloc[0]
    # 计算样本外 day_span_list 日对应的日期
    in_sample_date_list = date_episode_value_df.index[date_episode_value_df.index < in_sample_date_line]
    if len(in_sample_date_list) == 0:
        raise ValueError(f"in_sample_date_line={date_2_str(in_sample_date_line)} 没有样本内数据")
    baseline_date = in_sample_date_list[-1]
    idx, idx_max = len(in_sample_date_list) - 1, date_episode_value_df.shape[0] - 1
    ndays_date_dic = {n_days: date_episode_value_df.index[idx + n_days]
                      for n_days in day_span_list if idx + n_days <= idx_max}
    # 样本内数据最后一天的 close 净值
    baseline_net_close = date_episode_value_df['close'].loc[baseline_date]
    for n_days, cur_date in ndays_date_dic.items():
        cur_date = ndays_date_dic[n_days]
        cur_date_net_close = date_episode_value_df['close'].loc[cur_date]
        title = f'{title_header}_episode_value_[{n_days}]_{date_2_str(cur_date)}'
        compare_value_df = date_episode_value_df[episode_list].loc[[cur_date, baseline_date]].T
        baseline_df = pd.DataFrame([[cur_date_net_close, baseline_net_close] for _ in episode_list],
                                   index=episode_list,
                                   columns=[f'{date_2_str(cur_date)}_close', f'{date_2_str(baseline_net_close)}_close'])
        gap_s = (compare_value_df[cur_date] - compare_value_df[baseline_date]) * 100
        gap_s.name = 'gap*100'
        file_path = plot_twin([compare_value_df, baseline_df], gap_s,
                              name=title, y_scales_log=[False, False], **enable_kwargs)
        result_dic.setdefault('episode_value_compare', {})[n_days] = file_path

    # 随 Episode 增长，value 结果变化曲线
    episode_list.sort()
    episode_value_df = pd.DataFrame({episode: calc_reward_nav_value(reward_df, in_sample_date_line)
                                     for episode, reward_df in episode_reward_df_dic.items()
                                     if reward_df.shape[0] > 0}).T.sort_index()
    # 将 value 净值化，以方便与 close 进行比对
    result_dic['episode_trend_in_sample_summary_df'] = episode_value_df
    title = f'{title_header}_episode_trend_in_sample_summary'
    file_path = plot_twin([episode_value_df[['value', 'value_fee0']], episode_value_df['close']],
                          episode_value_df['action_count'],
                          # folder_path=cache_folder_path,
                          name=title, y_scales_log=[False, True], **enable_kwargs)
    # logger.debug("predict_result_df=\n%s", predict_result_df)
    result_dic['episode_trend_in_sample_summary_plot'] = file_path

    # 各个 episod 样本外 5日、10日、20日，60日、120日收益率 变化
    episode_value_dic = {episode: calc_reward_nav_value(reward_df)
                         for episode, reward_df in episode_reward_df_dic.items() if reward_df.shape[0] > 0}
    episode_value_df = pd.DataFrame(episode_value_dic).T.sort_index()

    # 计算 样本外 'value' 近5日、10日、20日，60日、120日收益率、年华收益率
    if in_sample_date_line is not None:
        days_rr_dic = {}
        # 日期索引，episod 列名，记录 value 值的 DataFrame
        date_episode_value_df = pd.DataFrame({episode: reward_df['value']
                                              for episode, reward_df in episode_reward_df_dic.items()
                                              if reward_df.shape[0] > 0})
        idx_max = date_episode_value_df.shape[0] - 1
        baseline_date = max(date_episode_value_df.index[date_episode_value_df.index <= in_sample_date_line])
        date_line_index = np.argmax(date_episode_value_df.index == baseline_date)
        date_line_value_s = date_episode_value_df.loc[baseline_date, :]
        for n_days in day_span_list:
            idx = date_line_index + n_days
            if idx <= idx_max:
                date_curr = date_episode_value_df.index[idx]
                rr_s = date_episode_value_df.iloc[idx, :] / date_line_value_s
                days_rr_dic[f'{n_days}_rr'] = rr_s - 1
                days_rr_dic[f'{n_days}_cagr'] = rr_s ** 365 / (date_curr - baseline_date).days
        episode_rr_df = pd.DataFrame(days_rr_dic)
        episode_value_df = pd.merge(episode_value_df, episode_rr_df, left_index=True, right_index=True)

    title = f'{title_header}_episode_trend_summary'
    file_path = plot_twin([episode_value_df[['value', 'value_fee0']], episode_value_df[['close']]],
                          episode_value_df['action_count'],
                          # folder_path=cache_folder_path,
                          name=title, y_scales_log=[False, True], **enable_kwargs)
    # logger.debug("predict_result_df=\n%s", predict_result_df)
    result_dic['episode_trend_summary_plot'] = file_path
    result_dic['episode_trend_summary_df'] = episode_value_df

    # value，value_fee0 走势图
    # 每一个 reward 一张图
    if show_plot_141:
        # smaller_kwargs = enable_kwargs.copy()
        # smaller_kwargs['figsize'] = (4.8, 6.4)
        value_plot_dic, episode_reward_dic = {}, {}
        for num, episode in enumerate(episode_list):
            if episode_reward_df_dic[episode].shape[0] == 0:
                continue
            reward_df = episode_reward_df_dic[episode]
            if in_sample_date_line is not None:
                baseline_date = max(reward_df.index[reward_df.index <= in_sample_date_line])
                if reward_df.loc[baseline_date, 'value'] < reward_df.iloc[0]['value']:
                    continue
            value_df = pd.DataFrame({f'{episode}_v': reward_df['value'],
                                     f'{episode}_0': reward_df['value_fee0']})

            # 例如：d3qn_in_sample_205-01-01_r0_2019-09-10
            title = f'{title_header}_value_{episode}'
            file_path = plot_twin([value_df[f'{episode}_v'], value_df[f'{episode}_0']], close_s,
                                  name=title,
                                  # folder_path=cache_folder_path,
                                  in_sample_date_line=in_sample_date_line, **enable_kwargs)
            value_plot_dic[episode] = file_path
            episode_reward_dic[episode] = episode_reward_df_dic[episode].iloc[-1:, :]

        result_dic['value_plot_dic'] = value_plot_dic
        result_dic['episode_reward_dic'] = episode_reward_dic

    # value，value_fee0 走势图
    # 合并展示图
    from ibats_utils.mess import split_chunk
    line_count = 4
    for episode_list_sub in split_chunk(
            [episode for episode in episode_list if episode_reward_df_dic[episode].shape[0] > 0], line_count):
        value_df = pd.DataFrame({f'{episode}_v': episode_reward_df_dic[episode]['value']
                                 for num, episode in enumerate(episode_list_sub)})
        value_fee0_df = pd.DataFrame({f'{episode}_0': episode_reward_df_dic[episode]['value_fee0']
                                      for num, episode in enumerate(episode_list_sub)})

        # 例如：d3qn_in_sample_205-01-01_r0_2019-09-10
        title = f"{title_header}_value_tot_[{min(episode_list_sub)}-{max(episode_list_sub)}]"
        file_path = plot_twin([value_df, value_fee0_df], close_s, name=title,
                              # folder_path=cache_folder_path,
                              in_sample_date_line=in_sample_date_line, **enable_kwargs)
        result_dic.setdefault('value_plot_list', []).append(file_path)

    result_dic['episode_reward_df'] = pd.DataFrame(
        {episode: df.iloc[-1, :] for episode, df in episode_reward_df_dic.items()}
    ).T.sort_index()

    # in_sample_date_line 节点后绩效分析
    perfomance_dic, has_close = {}, False
    for num, (episode, reward_df) in enumerate(episode_reward_df_dic.items()):
        if reward_df.shape[0] == 0:
            continue
        if in_sample_date_line is not None:
            df = reward_df[reward_df.index >= in_sample_date_line]
        else:
            df = reward_df

        if df.shape[0] == 0:
            continue
        df = df[['value', 'value_fee0']] / df[['value', 'value_fee0']].iloc[0, :]
        df.rename(columns={'value': f'{episode}_value', 'value_fee0': f'{episode}_value_fee0'}, inplace=True)
        if not has_close:
            df = pd.merge(close_s.loc[df.index], df, left_index=True, right_index=True)
            has_close = True

        # 绩效分析
        try:
            stats_result = df.calc_stats()
            stats_result.set_riskfree_rate(risk_free)
            perfomance_dic.update({_: stats.stats for _, stats in stats_result.items()})
        except:
            logger.exception(f"df.calc_stats() exception, df.shape={df.shape}, df.column={df.columns}\n")
            pass
    if len(perfomance_dic) > 0:
        result_dic['stats_df'] = pd.DataFrame(perfomance_dic)

    return result_dic


def _test_analysis_rewards_with_md(auto_open_file=True):
    # 读取 reward csv 文件
    folder_path = r'/home/mg/github/code_mess/drl/drl_off_example/d3qn_replay_2019_08_25/output/2013-05-13/model'
    target_round_n = [0]
    episode_reward_df_dic = {}
    index_col = ['trade_date']
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isdir(file_path):
            continue
        file_name_no_extension, extension = os.path.splitext(file_name)
        if extension != '.csv':
            continue
        # _, file_name_no_extension = os.path.split(file_name)
        # model_path = f"model/weights_1.h5"
        _, round_n, episode = file_name_no_extension.split('_')
        if int(round_n) != target_round_n:
            continue
        episode = int(episode)
        episode_reward_df_dic[episode] = pd.read_csv(file_path, index_col=index_col, parse_dates=index_col)

    in_sample_date_line = '2013-05-13'
    title_header = f'd3qn_replay_2019_08_25_{in_sample_date_line}_{target_round_n}'
    # 建立相关数据
    from ibats_common.example.data import load_data
    md_df = load_data('RB.csv',
                      folder_path=DATA_FOLDER_PATH, index_col='trade_date'
                      )
    param_dic = {}
    result_dic = analysis_rewards_with_md(episode_reward_df_dic, md_df, title_header,
                                          in_sample_date_line=in_sample_date_line, show_plot_141=True)
    file_path = summary_rewards_2_docx(param_dic, result_dic, title_header)
    logger.debug('文件路径：%s', file_path)
    if auto_open_file and file_path is not None:
        open_file_with_system_app(file_path)

    return file_path


if __name__ == "__main__":
    _test_analysis_rewards_with_md()
