#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 19-9-11 下午5:22
@File    : analysis.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
import itertools
import logging
import math
import os

import ffn
import numpy as np
import pandas as pd
from ibats_common.analysis.plot import plot_twin
from ibats_utils.mess import open_file_with_system_app, date_2_str

from analysis.summary import in_out_example_analysis_result_2_docx
from drl import DATA_FOLDER_PATH

logger = logging.getLogger(__name__)
logger.debug('import %s', ffn)


def analysis_rewards_with_md(episode_reward_df_dic, md_df, title_header, in_sample_date_line, enable_show_plot=False,
                             enable_save_plot=True, show_plot_141=False, risk_free=0.03, in_sample_only=False,
                             **kwargs):
    """分析 rewards 绩效"""
    # 整理参数
    # cache_folder_path = r'/home/mg/github/code_mess/output/cache'
    analysis_result_dic = {}
    day_span_list = [5, 10, 20, 60]
    close_s = md_df['close']
    episode_count, episode_list = len(episode_reward_df_dic), list(episode_reward_df_dic.keys())
    if episode_count == 0:
        return analysis_result_dic
    episode_list.sort()
    enable_kwargs = dict(enable_save_plot=enable_save_plot, enable_show_plot=enable_show_plot, figsize=(5.4, 5.0))
    in_sample_date_line = pd.to_datetime(in_sample_date_line)

    def calc_reward_nav_value(reward_df: pd.DataFrame, baseline=None):
        df = reward_df[['value', 'value_fee0', 'close', 'action_count']]
        avg_holding_s = df.shape[0] / df['action_count'] * 2
        # 部分无效数据奇高导致曲线不变不明显，因此，对超过阈值的数据进行一定的处理
        threshold = 20
        is_fit = avg_holding_s > threshold
        avg_holding_s[is_fit] = threshold + np.log(avg_holding_s[is_fit] - threshold + 1)
        df['avg_holding'] = avg_holding_s
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
    if not in_sample_only:
        ndays_date_dic = {n_days: date_episode_value_df.index[idx + n_days]
                          for n_days in day_span_list if idx + n_days <= idx_max}
        # 样本内数据最后一天的 close 净值
        baseline_net_close = date_episode_value_df['close'].loc[baseline_date]
        for n_days, cur_date in ndays_date_dic.items():
            cur_date = ndays_date_dic[n_days]
            cur_date_net_close = date_episode_value_df['close'].loc[cur_date]
            title = f'{title_header}_episode_value_[{n_days}]_{date_2_str(cur_date)}'
            compare_value_df = date_episode_value_df[episode_list].loc[[cur_date, baseline_date]].T
            baseline_df = pd.DataFrame(
                [[cur_date_net_close, baseline_net_close] for _ in episode_list],
                index=episode_list,
                columns=[f'{date_2_str(cur_date)}_close', f'{date_2_str(baseline_net_close)}_close'])
            gap_s = (compare_value_df[cur_date] - compare_value_df[baseline_date]) * 100
            gap_s.name = 'gap*100'
            file_path = plot_twin([compare_value_df, baseline_df], gap_s,
                                  name=title, y_scales_log=[False, False], **enable_kwargs)
            analysis_result_dic.setdefault('episode_value_compare', {})[n_days] = file_path

    # 随 Episode 增长，value 结果变化曲线
    episode_list.sort()
    episode_in_sample_value_df = pd.DataFrame({episode: calc_reward_nav_value(reward_df, in_sample_date_line)
                                               for episode, reward_df in episode_reward_df_dic.items()
                                               if reward_df.shape[0] > 0}).T.sort_index()

    # 筛选出有效的 模型
    def check_available_reward(episode, reward_df):
        """筛选出有效的 模型"""
        if episode < 500:
            return False
        df_len = reward_df.shape[0]
        if df_len == 0:
            return False
        last_s, first_s = reward_df.iloc[-1, :], reward_df.iloc[0, :]
        if last_s['nav'] <= 1.0:
            return False
        if last_s['action_count'] <= 0:
            return False
        avg_holding = df_len / last_s['action_count'] * 2
        if avg_holding <= 3 or 10 <= avg_holding:
            return False
        # 卡玛比 大于 1
        nav = reward_df['nav']
        cal_mar_threshold = 2.0
        cal_mar = nav.calc_calmar_ratio()
        if np.isnan(cal_mar) or cal_mar <= cal_mar_threshold:
            return False
        # 复合年华收益率大于 0.05
        # cagr = nav.calc_cagr()
        # if np.isnan(cagr) or cagr < 0.05:
        #     return False
        return True

    episode_value_pair_list = [(episode, reward_df['value'].iloc[-1])
                               for episode, reward_df in episode_reward_df_dic.items()
                               if check_available_reward(episode, reward_df)]
    episode_value_pair_list_len = len(episode_value_pair_list)
    # 对于筛选后，有效模型数量过多的情况，采取剔除前、后20%的方式缩减数量的同时，剔除极端数据
    if episode_value_pair_list_len >= 10:
        # 剔除分位数前 20%，以及后20%的数字
        episode_value_pair_list.sort(key=lambda x: x[1])
        episode_value_pair_list = episode_value_pair_list[
                                  math.ceil(episode_value_pair_list_len * 0.2):
                                  math.floor(episode_value_pair_list_len * 0.8)]
    available_episode_list = [_[0] for _ in episode_value_pair_list]
    available_episode_list.sort()
    analysis_result_dic['available_episode_list'] = available_episode_list
    episode_model_path_dic = kwargs.setdefault('episode_model_path_dic', None)
    if episode_model_path_dic is not None:
        analysis_result_dic['available_episode_model_path_dic'] = {
            episode: episode_model_path_dic[episode] for episode in available_episode_list}

    # 将 value 净值化，以方便与 close 进行比对
    analysis_result_dic['episode_in_sample_value_df'] = episode_in_sample_value_df
    title = f'{title_header}_episode_in_sample_value'
    file_path = plot_twin([episode_in_sample_value_df[['value', 'value_fee0']], episode_in_sample_value_df['close']],
                          episode_in_sample_value_df['avg_holding'],
                          # folder_path=cache_folder_path,
                          name=title, y_scales_log=[False, False], **enable_kwargs)
    # logger.debug("predict_result_df=\n%s", predict_result_df)
    analysis_result_dic['episode_in_sample_value_plot'] = file_path

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

    title = f'{title_header}_episode_value'
    file_path = plot_twin([episode_value_df[['value', 'value_fee0']], episode_value_df[['close']]],
                          episode_value_df['avg_holding'],
                          # folder_path=cache_folder_path,
                          name=title, y_scales_log=[False, False], **enable_kwargs)
    # logger.debug("predict_result_df=\n%s", predict_result_df)
    analysis_result_dic['episode_value_plot'] = file_path
    analysis_result_dic['episode_value_df'] = episode_value_df

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

        analysis_result_dic['value_plot_dic'] = value_plot_dic
        analysis_result_dic['episode_reward_dic'] = episode_reward_dic

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
        analysis_result_dic.setdefault('value_plot_list', []).append(file_path)

    analysis_result_dic['episode_reward_df'] = pd.DataFrame(
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
        analysis_result_dic['stats_df'] = pd.DataFrame(perfomance_dic)

    return analysis_result_dic


def _test_analysis_rewards_with_md(auto_open_file=True):
    from analysis.summary import summary_rewards_2_docx
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
    analysis_result_dic = analysis_rewards_with_md(episode_reward_df_dic, md_df, title_header,
                                                   in_sample_date_line=in_sample_date_line, show_plot_141=True)
    file_path = summary_rewards_2_docx(param_dic, analysis_result_dic, title_header)
    logger.debug('文件路径：%s', file_path)
    if auto_open_file and file_path is not None:
        open_file_with_system_app(file_path)

    return file_path


def calc_reward_nav_value(reward_df: pd.DataFrame, baseline=None):
    """根据 reward_df 返回 净值，0手续费净值，平均持仓，以及对应的收盘价，供输出plot使用"""
    if baseline is not None:
        df = reward_df[reward_df.index <= baseline]
    else:
        df = reward_df
    ret_s = df.iloc[-1, :][['nav', 'nav_fee0', 'close', 'action_count']].copy()
    ret_s['close'] /= df.iloc[0, :]['close']
    # 部分无效数据奇高导致曲线变化不明显，因此，对超过阈值的数据进行一定的处理
    threshold = 20
    avg_holding = df.shape[0] / ret_s['action_count'] * 2
    if avg_holding > threshold:
        avg_holding = threshold + np.log(avg_holding - threshold + 1)
    ret_s['avg_holding'] = avg_holding
    return ret_s


def check_available_reward(episode, reward_df):
    """筛选出有效的 模型"""
    if episode < 20:
        return False
    df_len = reward_df.shape[0]
    if df_len == 0:
        return False
    last_s, first_s = reward_df.iloc[-1, :], reward_df.iloc[0, :]
    if last_s['nav'] <= 1:
        return False
    if last_s['action_count'] <= 0:
        return False
    avg_holding = df_len / last_s['action_count'] * 2
    if avg_holding < 3 or 10 < avg_holding:
        return False
    # 卡玛比 大于 1
    nav = reward_df['nav']
    cal_mar_threshold = 1.0
    cal_mar = nav.calc_calmar_ratio()
    if np.isnan(cal_mar) or cal_mar <= cal_mar_threshold:
        return False
    # 复合年华收益率大于 0.05
    # cagr = nav.calc_cagr()
    # if np.isnan(cagr) or cagr < 0.05:
    #     return False
    return True


def analysis_in_out_example_valid_env_result(
        in_out_example_valid_env_result_dic, model_param_dic, in_sample_date_line, episode_model_path_dic=None,
        enable_show_plot=False, enable_save_plot=True, enable_2_docx=True,
        round_n=0, show_plot_141=False, risk_free=0.03, available_episode_num_filter=0, **kwargs):
    """对 样本内测试/样本外测试结果 及环境信息进行分析，并形成绩效分析报告"""

    # 整理参数
    analysis_result_dic = {}
    day_span_list = [5, 10, 20, 30, 60]
    in_sample_date_line = pd.to_datetime(in_sample_date_line)
    in_sample_date_line_str = date_2_str(in_sample_date_line)
    model_name = model_param_dic['model_name']
    title_header = f"{model_name}_{in_sample_date_line_str}_{round_n}"
    episode_reward_df_dic_key = 'episode_reward_df_dic'
    off_example_available_days = kwargs.setdefault('off_example_available_days', 20)
    for in_off_key in ('in_example', 'off_example'):
        if in_off_key not in in_out_example_valid_env_result_dic or \
                episode_reward_df_dic_key not in in_out_example_valid_env_result_dic[in_off_key]:
            continue
        analysis_result_dic[in_off_key] = analysis_result_dic_tmp = {}
        episode_reward_df_dic = in_out_example_valid_env_result_dic[in_off_key][episode_reward_df_dic_key]
        if episode_reward_df_dic is None or len(episode_reward_df_dic) == 0:
            logger.warning("基准日期 %s round=%d %s episode_reward_df_dic is None or length is 0",
                           in_sample_date_line_str, round_n, in_off_key)
            continue
        md_df = in_out_example_valid_env_result_dic[in_off_key]['md_df']
        close_s = md_df['close']
        episode_list = [_ for _ in episode_reward_df_dic.keys() if episode_reward_df_dic[_].shape[0] > 0]
        episode_count = len(episode_list),
        if episode_count == 0:
            continue
        episode_list.sort()
        enable_kwargs = dict(enable_save_plot=enable_save_plot, enable_show_plot=enable_show_plot, figsize=(5.4, 5.0))

        # 筛选有效的模型保存到 available_episode_list，并将模型路经保存到 available_episode_model_path_dic
        episode_nav_pair_list = [(
            episode, reward_df['nav'].iloc[-1] if in_off_key == 'in_example' else
            reward_df['nav'].iloc[-1 if reward_df.shape[0] < off_example_available_days else off_example_available_days]
        )
            for episode, reward_df in episode_reward_df_dic.items() if check_available_reward(episode, reward_df)
        ]
        episode_nav_pair_list_len = len(episode_nav_pair_list)
        if episode_nav_pair_list_len == 0:
            logger.warning("基准日期 %s round=%d %s 共 %d 个 Episode，没有找到有效的模型",
                           in_sample_date_line_str, round_n, in_off_key, len(episode_reward_df_dic))
        else:
            logger.info("基准日期 %s round=%d %s 共 %d 个 Episode，有效的模型 %d 个",
                        in_sample_date_line_str, round_n, in_off_key, len(episode_reward_df_dic),
                        episode_nav_pair_list_len)

        # 对于筛选后，有效模型数量过多的情况，采取剔除前、后20%的方式缩减数量的同时，剔除极端数据
        if available_episode_num_filter is not None and episode_nav_pair_list_len >= available_episode_num_filter:
            # 剔除分位数前 20%，以及后20%的数字
            episode_nav_pair_list.sort(key=lambda x: x[1])
            episode_nav_pair_list = episode_nav_pair_list[
                                    math.ceil(episode_nav_pair_list_len * 0.2):
                                    math.floor(episode_nav_pair_list_len * 0.8)]
        available_episode_list = [_[0] for _ in episode_nav_pair_list]
        available_episode_list.sort()
        analysis_result_dic_tmp['available_episode_list'] = available_episode_list
        if episode_model_path_dic is not None:
            analysis_result_dic_tmp['available_episode_model_path_dic'] = {
                episode: episode_model_path_dic[episode] for episode in available_episode_list}

        # 随 Episode 增长，nav 结果变化曲线
        # 输出文件保存到 episode_in_sample_value_plot
        n_days_file_path_dic = {}
        for n_days in itertools.chain(day_span_list, [-1]):
            if in_off_key == 'in_example' and n_days != -1:
                # 样本内测试的情况下，值看最后的净值
                continue
            episode_nav_df = pd.DataFrame({
                episode: calc_reward_nav_value(reward_df.iloc[:n_days])
                for episode, reward_df in episode_reward_df_dic.items() if reward_df.shape[0] > 0}
            ).T.sort_index()
            if episode_nav_df.shape[0] == 0:
                continue
            title = f'{title_header}_{in_off_key}_episode_nav_plot_{n_days if n_days > 0 else ""}'
            try:
                file_path = plot_twin([episode_nav_df[['nav', 'nav_fee0']], episode_nav_df['close']],
                                      episode_nav_df['avg_holding'],
                                      name=title, y_scales_log=[False, False], **enable_kwargs)
            except KeyError:
                logger.exception("episode_nav_df,shape=%s, episode_nav_df.head():\n%s\n",
                                 episode_nav_df.shape, episode_nav_df.head())
                continue
            n_days_file_path_dic[n_days] = file_path
        analysis_result_dic_tmp['episode_nav_plot_file_path_dic'] = n_days_file_path_dic

        # 各个 episode 样本外 5日、10日、20日，60日、120日收益率 变化
        episode_nav_df = pd.DataFrame({
            episode: calc_reward_nav_value(reward_df)
            for episode, reward_df in episode_reward_df_dic.items() if reward_df.shape[0] > 0}
        ).T.sort_index()
        if episode_nav_df.shape[0] == 0:
            logger.warning("round=%d %s episode_reward_df_dic.keys=%s, episode_nav_df.shape[0] == 0",
                           round_n, in_off_key, episode_reward_df_dic.keys())
            continue
        # 日期索引，episode 列名，记录 value 值的 DataFrame
        date_episode_nav_df = pd.DataFrame({
            episode: reward_df['nav']
            for episode, reward_df in episode_reward_df_dic.items()
            if reward_df.shape[0] > 0}).sort_index()
        data_count = date_episode_nav_df.shape[0]
        if data_count == 0:
            logger.warning("round=%d %s episode_reward_df_dic.keys=%s\ndate_episode_nav_df.shape[0] == 0",
                           round_n, in_off_key, episode_reward_df_dic.keys())
            continue
        baseline_date = date_episode_nav_df.index[0]
        for n_days in day_span_list:
            if in_off_key == 'in_example':
                continue
            if data_count <= n_days:
                continue
            date_curr = date_episode_nav_df.index[n_days]
            nav_s = date_episode_nav_df.iloc[n_days, :]
            days_rr_dic = {
                f'{n_days}_rr': nav_s - 1,
                f'{n_days}_cagr': nav_s ** (365 / (date_curr - baseline_date).days) - 1
            }
            episode_rr_df = pd.DataFrame(days_rr_dic)
            episode_nav_df = pd.merge(episode_nav_df, episode_rr_df, left_index=True, right_index=True)

        # 记录随 episode 增长，'nav', 'nav_fee0', 'avg_holding', 'close',
        # 以及 f'{n_days}_rr', f'{n_days}_cagr' 变化
        analysis_result_dic_tmp['episode_nav_df'] = episode_nav_df

        # nav，nav 走势图
        # 每一个 reward 一张图
        if show_plot_141:
            # smaller_kwargs = enable_kwargs.copy()
            # smaller_kwargs['figsize'] = (4.8, 6.4)
            episode_nav_plot_path_dic, episode_reward_dic = {}, {}
            for num, episode in enumerate(episode_list):
                reward_df = episode_reward_df_dic[episode]
                nav_df = pd.DataFrame({f'{episode}_v': reward_df['nav'],
                                       f'{episode}_0': reward_df['nav_fee0']})

                # 例如：d3qn_in_sample_205-01-01_r0_2019-09-10
                title = f'{title_header}_{in_off_key}_nav_{episode}'
                file_path = plot_twin([nav_df[f'{episode}_v'], nav_df[f'{episode}_0']], close_s,
                                      name=title,
                                      # folder_path=cache_folder_path,
                                      in_sample_date_line=in_sample_date_line, **enable_kwargs)
                episode_nav_plot_path_dic[episode] = file_path

            analysis_result_dic_tmp['episode_nav_plot_path_dic'] = episode_nav_plot_path_dic

        # nav，nav_fee0 走势图
        # 合并展示图
        from ibats_utils.mess import split_chunk
        line_count = 4
        for episode_list_sub in split_chunk(episode_list, line_count):
            nav_df = pd.DataFrame({f'{episode}_v': episode_reward_df_dic[episode]['nav']
                                   for num, episode in enumerate(episode_list_sub)})
            nav_fee0_df = pd.DataFrame({f'{episode}_0': episode_reward_df_dic[episode]['nav_fee0']
                                        for num, episode in enumerate(episode_list_sub)})

            # 例如：d3qn_in_sample_205-01-01_r0_2019-09-10
            title = f"{title_header}_{in_off_key}_nav_episode[{min(episode_list_sub)}-{max(episode_list_sub)}]"
            file_path = plot_twin([nav_df, nav_fee0_df], close_s, name=title,
                                  # folder_path=cache_folder_path,
                                  in_sample_date_line=in_sample_date_line, **enable_kwargs)
            analysis_result_dic_tmp.setdefault('nav_plot_file_path_list', []).append(file_path)

        analysis_result_dic_tmp['episode_reward_df'] = pd.DataFrame(
            {episode: df.iloc[-1, :] for episode, df in episode_reward_df_dic.items()}
        ).T.sort_index()

        # 绩效分析
        performance_dic, has_close = {}, False
        for num, episode in enumerate(episode_list):
            reward_df = episode_reward_df_dic[episode]
            reward_df.rename(columns={'nav': f'{episode}_nav', 'nav_fee0': f'{episode}_nav_fee0'}, inplace=True)
            if not has_close:
                reward_df = pd.merge(close_s.loc[reward_df.index], reward_df, left_index=True, right_index=True)
                has_close = True

            # 绩效分析
            try:
                stats_result = reward_df.calc_stats()
                stats_result.set_riskfree_rate(risk_free)
                performance_dic.update({_: stats.stats for _, stats in stats_result.items()})
            except:
                logger.exception(
                    f"df.calc_stats() exception, df.shape={reward_df.shape}, df.column={reward_df.columns}\n")
                pass
        if len(performance_dic) > 0:
            analysis_result_dic_tmp['stats_df'] = pd.DataFrame(performance_dic)

    if enable_2_docx:
        model_param_dic['round_n'] = round_n
        model_param_dic['in_sample_date_line'] = in_sample_date_line
        summary_file_path = in_out_example_analysis_result_2_docx(
            model_param_dic, analysis_result_dic, title_header, in_sample_date_line=in_sample_date_line)
        logger.debug('summary to docx [%d] %s', round_n, summary_file_path)
    else:
        summary_file_path = None

    return analysis_result_dic, summary_file_path


def _test_analysis_in_out_example_valid_env_result():
    import functools
    from collections import defaultdict
    from ibats_common.backend.factor import transfer_2_batch, get_factor
    from ibats_common.example.data import OHLCAV_COL_NAME_LIST, load_data
    from drl.d3qn_r_2019_10_11.agent.main import MODEL_NAME
    instrument_type = 'RB'
    in_sample_date_line = '2017-01-26'
    model_folder = f'/home/mg/github/code_mess/drl/d3qn_r_2019_10_11/output/{in_sample_date_line}/model'
    analysis_kwargs_round_n = 1
    target_round_n_list = [analysis_kwargs_round_n]
    max_valid_data_len = 1000
    n_step = 60
    index_col = ['trade_date']

    logger.debug('准备生成数据')
    from ibats_common.example import get_trade_date_series
    trade_date_series = get_trade_date_series(DATA_FOLDER_PATH)
    from ibats_common.example import get_delivery_date_series
    delivery_date_series = get_delivery_date_series(instrument_type, DATA_FOLDER_PATH)
    get_factor_func = functools.partial(get_factor,
                                        trade_date_series=trade_date_series, delivery_date_series=delivery_date_series)
    md_loader_func = lambda range_to=None: load_data(
        f'{instrument_type}.csv', folder_path=DATA_FOLDER_PATH, index_col='trade_date', range_to=range_to
    )[OHLCAV_COL_NAME_LIST]

    # 加载模型列表[round_n][episode] = model_file_path
    round_n_episode_model_path_dic = defaultdict(lambda: defaultdict(str))
    for file_name in os.listdir(model_folder):
        model_file_path = os.path.join(model_folder, file_name)
        if os.path.isdir(model_file_path):
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
        round_n_episode_model_path_dic[round_n][episode] = model_file_path

    episode_model_path_dic = round_n_episode_model_path_dic[analysis_kwargs_round_n]
    analysis_kwargs = dict()
    analysis_kwargs['in_sample_date_line'] = date_2_str(in_sample_date_line)
    analysis_kwargs['round_n'] = analysis_kwargs_round_n
    analysis_kwargs['episode_model_path_dic'] = episode_model_path_dic
    analysis_kwargs['model_param_dic'] = dict(
        model_name=MODEL_NAME,
        n_step=n_step,
        model_folder=model_folder,
    )
    analysis_kwargs['off_example_available_days'] = 20
    in_out_example_valid_env_result_dic = {}
    for key in ('in_example', 'off_example'):
        md_df = md_loader_func()
        md_df.index = pd.DatetimeIndex(md_df.index)

        if key == 'off_example':
            date_from = pd.to_datetime(in_sample_date_line) + pd.DateOffset(1)
        else:
            if max_valid_data_len is not None and \
                    md_df[md_df.index <= pd.to_datetime(in_sample_date_line)].shape[0] > max_valid_data_len > 0:
                date_from = min(md_df.sort_index().index[-max_valid_data_len:])
            else:
                date_from = None
        # 生成因子
        factors_df = get_factor_func(md_df)
        df_index, df_columns, batch_factors = transfer_2_batch(factors_df, n_step=n_step,
                                                               date_from=date_from)
        logger.info('batch_factors.shape=%s', batch_factors.shape)
        md_df = md_df.loc[df_index, :]

        csv_file_name_key = '_off' if key == 'off_example' else '_in'
        episode_reward_df_dic = {}
        episode_list = list(episode_model_path_dic.keys())
        episode_list.sort()
        episode_count = len(episode_list)
        logger.debug('%s round %d has %d episode', key, analysis_kwargs_round_n, episode_count)
        for num, episode in enumerate(episode_list, start=1):
            model_file_path = str(episode_model_path_dic[episode])
            reward_file_name = \
                f'reward_{analysis_kwargs_round_n}_{episode}{csv_file_name_key if csv_file_name_key is not None else ""}.csv'
            model_folder, _ = os.path.split(model_file_path)
            reward_file_path = os.path.join(model_folder, reward_file_name)
            if os.path.exists(reward_file_path):
                reward_df = pd.read_csv(reward_file_path, index_col=index_col, parse_dates=index_col)
                if reward_df.shape[0] == 0:
                    continue
                episode_reward_df_dic[episode] = reward_df

        in_out_example_valid_env_result_dic[key] = dict(
            episode_reward_df_dic=episode_reward_df_dic,
            md_df=md_df
        )

    enable_summary_rewards_2_docx = True
    logger.debug('准备调用函数')
    analysis_result_dic, round_n_summary_file_path = analysis_in_out_example_valid_env_result(
        in_out_example_valid_env_result_dic, enable_2_docx=enable_summary_rewards_2_docx, **analysis_kwargs)
    for key, value in analysis_result_dic.items():
        logger.debug("analysis_result_dic[%s].keys():\n%s", key, analysis_result_dic[key].keys())


if __name__ == "__main__":
    # _test_analysis_rewards_with_md()
    _test_analysis_in_out_example_valid_env_result()
