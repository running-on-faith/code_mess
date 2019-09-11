#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 19-9-6 上午7:24
@File    : summary.py
@contact : mmmaaaggg@163.com
@desc    : 
"""

import datetime
import logging
import os

import docx
import ffn
import pandas as pd
import numpy as np
from docx.shared import Pt
from ibats_common.backend.mess import get_report_folder_path
from ibats_common.analysis.plot import clean_cache
from ibats_common.analysis.plot import plot_twin
from ibats_common.analysis.summary import df_2_table, stats_df_2_docx_table
from ibats_utils.mess import datetime_2_str, open_file_with_system_app, date_2_str

from drl import DATA_FOLDER_PATH

logger = logging.getLogger(__name__)
logger.debug('import %s', ffn)
STR_FORMAT_DATETIME_4_FILE_NAME = '%Y-%m-%d %H_%M_%S'
FORMAT_2_PERCENT = lambda x: f"{x * 100: .2f}%"
FORMAT_2_FLOAT2 = r"{0:.2f}"
FORMAT_2_FLOAT4 = r"{0:.4f}"
FORMAT_2_MONEY = r"{0:,.2f}"


def analysis_rewards_with_md(episode_reward_df_dic, md_df, title_header, enable_show_plot=False,
                             enable_save_plot=True, show_plot_141=False,
                             in_sample_date_line=None, risk_free=0.03,
                             **kwargs):
    """分析 rewards 绩效"""
    cache_folder_path = r'/home/mg/github/code_mess/output/cache'
    result_dic = {}
    episode_count, episode_list = len(episode_reward_df_dic), list(episode_reward_df_dic.keys())
    if episode_count == 0:
        return result_dic

    # 整理参数
    enable_kwargs = dict(enable_save_plot=enable_save_plot, enable_show_plot=enable_show_plot, figsize=(6, 8))
    in_sample_date_line = None if in_sample_date_line is None else pd.to_datetime(in_sample_date_line)

    # 随 Episode 增长，value 结果变化曲线
    episode_list.sort()
    if in_sample_date_line is not None:
        in_sample_result_df = pd.DataFrame({_: val.loc[max(val.index[val.index <= in_sample_date_line])]
                                            for _, val in episode_reward_df_dic.items()}).T.sort_index()
        result_dic['episode_trend_in_sample_summary_df'] = in_sample_result_df
        title = f'{title_header}_episode_trend_in_sample_summary'
        file_path = plot_twin(in_sample_result_df[['value', 'value_fee0']], in_sample_result_df['action_count'],
                              folder_path=cache_folder_path,
                              name=title, y_scales_log=[False, True], **enable_kwargs)
        # logger.debug("predict_result_df=\n%s", predict_result_df)
        result_dic['episode_trend_in_sample_summary_plot'] = file_path

    # 各个 episod 最后 reward 值
    episode_value_dic = {episode: reward_df[['value', 'value_fee0', 'action_count']].iloc[-1, :]
                         for episode, reward_df in episode_reward_df_dic.items()}
    episode_value_df = pd.DataFrame(episode_value_dic).T.sort_index()

    # 计算 样本外 'value' 近5日、10日、20日，60日、120日收益率、年华收益率
    if in_sample_date_line is not None:
        days_rr_dic = {}
        # 日期索引，episod 列名，记录 value 值的 DataFrame
        date_episode_value_df = pd.DataFrame(
            {episode: reward_df['value'] for episode, reward_df in episode_reward_df_dic.items()})
        idx_max = date_episode_value_df.shape[0] - 1
        date_baseline = max(date_episode_value_df.index[date_episode_value_df.index <= in_sample_date_line])
        date_line_index = np.argmax(date_episode_value_df.index == date_baseline)
        date_line_value_s = date_episode_value_df.loc[date_baseline, :]
        for n_days in [5, 10, 20, 60, 120]:
            idx = date_line_index + n_days
            if idx <= idx_max:
                date_curr = date_episode_value_df.index[idx]
                rr_s = date_episode_value_df.iloc[idx, :] / date_line_value_s
                days_rr_dic[f'{n_days}_rr'] = rr_s - 1
                days_rr_dic[f'{n_days}_cagr'] = rr_s ** 365 / (date_curr - date_baseline).days
        episode_rr_df = pd.DataFrame(days_rr_dic)
        episode_value_df = pd.merge(episode_value_df, episode_rr_df, left_index=True, right_index=True)

    title = f'{title_header}_episode_trend_summary'
    file_path = plot_twin(episode_value_df[['value', 'value_fee0']], episode_value_df['action_count'],
                          folder_path=cache_folder_path,
                          name=title, y_scales_log=[False, True], **enable_kwargs)
    # logger.debug("predict_result_df=\n%s", predict_result_df)
    result_dic['episode_trend_summary_plot'] = file_path
    result_dic['episode_trend_summary_df'] = episode_value_df

    # value，value_fee0 走势图
    # 每一个 reward 一张图
    if show_plot_141:
        smaller_kwargs = enable_kwargs.copy()
        smaller_kwargs['figsize'] = (5.6, 6.8)
        value_plot_dic, episode_reward_dic = {}, {}
        for num, episode in enumerate(episode_list):
            if episode_reward_df_dic[episode].shape[0] == 0:
                continue
            reward_df = episode_reward_df_dic[episode]
            if in_sample_date_line is not None:
                date_baseline = max(reward_df.index[reward_df.index <= in_sample_date_line])
                if reward_df.loc[date_baseline, 'value'] < reward_df.iloc[0]['value']:
                    continue
            value_df = pd.DataFrame({f'{episode}_v': reward_df['value'],
                                     f'{episode}_0': reward_df['value_fee0']})

            # 例如：d3qn_in_sample_205-01-01_r0_2019-09-10
            title = f'{title_header}_value_{episode}'
            file_path = plot_twin([value_df[f'{episode}_v'], value_df[f'{episode}_0']], md_df['close'],
                                  name=title,
                                  folder_path=cache_folder_path,
                                  in_sample_date_line=in_sample_date_line, **smaller_kwargs)
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
        file_path = plot_twin([value_df, value_fee0_df], md_df['close'], name=title,
                              folder_path=cache_folder_path,
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
            df = pd.merge(md_df.loc[df.index, 'close'], df, left_index=True, right_index=True)
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


def summary_4_rewards(result_dic, md_df, title_header, enable_clean_cache=False,
                      doc_file_path=None, in_sample_date_line=None):
    """创建 绩效分析结果 word 文档"""
    # 整理参数
    in_sample_date_line = None if in_sample_date_line is None else pd.to_datetime(in_sample_date_line)
    in_sample_date_line_str = date_2_str(in_sample_date_line) if in_sample_date_line is not None else ""
    int_col_name_set = {'action', 'action_count'}
    money_col_name_set = {'value', 'cash', 'fee_tot', 'value_fee0'}
    available_reward_col_name_list = ['value', 'cash', 'fee_tot', 'value_fee0', 'action_count']

    # 生成 docx 文档将所需变量
    heading_title = f'Rewards 分析报告[{title_header}] '

    # 生成 docx 文件
    document = docx.Document()
    # 设置默认字体
    document.styles['Normal'].font.name = '微软雅黑'
    document.styles['Normal']._element.rPr.rFonts.set(docx.oxml.ns.qn('w:eastAsia'), '微软雅黑')
    # 创建自定义段落样式(第一个参数为样式名, 第二个参数为样式类型, 1为段落样式, 2为字符样式, 3为表格样式)
    UserStyle1 = document.styles.add_style('UserStyle1', 1)
    # 设置字体尺寸
    UserStyle1.font.size = docx.shared.Pt(40)
    # 设置字体颜色
    UserStyle1.font.color.rgb = docx.shared.RGBColor(0xff, 0xde, 0x00)
    # 居中文本
    UserStyle1.paragraph_format.alignment = docx.enum.text.WD_ALIGN_PARAGRAPH.CENTER
    # 设置中文字体
    UserStyle1.font.name = '微软雅黑'
    UserStyle1._element.rPr.rFonts.set(docx.oxml.ns.qn('w:eastAsia'), '微软雅黑')

    # 文件内容
    document.add_heading(heading_title, 0).alignment = docx.enum.text.WD_ALIGN_PARAGRAPH.CENTER
    document.add_paragraph('')
    document.add_paragraph('')
    heading_count, sub_heading_count = 1, 1
    document.add_heading(f'{heading_count}、展示训练曲线', 1)
    key = 'episode_trend_in_sample_summary_df'
    if key in result_dic:
        document.add_heading(f'{heading_count}.{sub_heading_count}、随 Episode 增长，value 结果变化曲线（样本内）'
                             f'{in_sample_date_line_str}', 2)
        document.add_picture(result_dic['episode_trend_in_sample_summary_plot'])  # , width=docx.shared.Inches(1.25)
        document.add_paragraph('')

    key = 'episode_trend_summary_df'
    if key in result_dic:
        sub_heading_count += 1
        document.add_heading(f'{heading_count}.{sub_heading_count}、随 Episode 增长，value 结果变化曲线', 2)
        document.add_picture(result_dic['episode_trend_summary_plot'])  # , width=docx.shared.Inches(1.25)
        document.add_paragraph('详细数据')
        data_df = result_dic[key]

        format_by_col = {_: FORMAT_2_MONEY for _ in data_df.columns if _ in money_col_name_set}
        format_by_col.update({_: FORMAT_2_PERCENT for _ in data_df.columns if _.find('rr') > 0 or _.find('cagr') > 0})
        df_2_table(document, data_df, format_by_col=format_by_col, max_col_count=5)
        heading_count += 1
        # 添加分页符
        document.add_page_break()

    # 展示 Value Plot
    document.add_heading(f'{heading_count}、Reward Value 曲线走势', 1)
    # 汇总展示
    sub_heading_count = 1
    document.add_heading(f'{heading_count}.{sub_heading_count}、合并绘图走势', 2)
    for num, file_path in enumerate(result_dic['value_plot_list'], start=1):
        document.add_paragraph(f'{heading_count}.{sub_heading_count}.{num}')
        document.add_picture(file_path)
        document.add_paragraph('')
    data_df = result_dic['episode_reward_df'][available_reward_col_name_list]
    format_by_col = {_: FORMAT_2_MONEY for _ in data_df.columns if _ in money_col_name_set}
    df_2_table(document, data_df, format_by_col=format_by_col, max_col_count=5,
               mark_top_n=5, mark_top_n_on_cols=data_df.columns)
    document.add_paragraph('')
    key = 'value_plot_dic'
    if key in result_dic:
        sub_heading_count += 1
        document.add_heading(f'{heading_count}.{sub_heading_count}、详细数据', 2)
        # 分别展示
        for num, (episode, file_path) in enumerate(result_dic[key].items(), start=2):
            document.add_heading(f'{heading_count}.{sub_heading_count}.{num}、episode={episode}', 2)
            document.add_picture(file_path)
            document.add_paragraph('')
            data_df = result_dic['episode_reward_dic'][episode][available_reward_col_name_list]
            format_by_col = {_: FORMAT_2_MONEY for _ in data_df.columns if _ in money_col_name_set}
            df_2_table(document, data_df, format_by_col=format_by_col, max_col_count=5)
            document.add_paragraph('')

    heading_count += 1
    document.add_page_break()

    document.add_heading(f'{heading_count}、样本外数据绩效统计数据（Porformance stat）', 1)
    key = 'stats_df'
    if key in result_dic:
        stats_df = result_dic[key].T
        stats_df.drop(['start', 'rf'], axis=1, inplace=True)
        stats_df_2_docx_table(stats_df, document, format_axis='column',
                              mark_top_n=5, mark_top_n_on_cols=[_ for _ in stats_df.columns if _ not in {'end'}])
        heading_count += 1
        document.add_page_break()

    # 保存文件
    if doc_file_path is not None:
        if os.path.isdir(doc_file_path):
            folder_path, file_name = doc_file_path, ''
        else:
            folder_path, file_name = os.path.split(doc_file_path)
    else:
        folder_path, file_name = None, ''

    if folder_path is None or folder_path == "":
        folder_path = get_report_folder_path()

    if folder_path != '' and not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if file_name == '':
        file_name = f"{title_header}" \
            f"{datetime_2_str(datetime.datetime.now(), STR_FORMAT_DATETIME_4_FILE_NAME)}.docx"
        file_path = os.path.join(folder_path, file_name)
    else:
        file_path = doc_file_path

    document.save(file_path)
    if enable_clean_cache:
        clean_cache()

    return file_path


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
    result_dic = analysis_rewards_with_md(episode_reward_df_dic, md_df, title_header,
                                          in_sample_date_line=in_sample_date_line, show_plot_141=True)
    file_path = summary_4_rewards(result_dic, md_df, title_header)
    logger.debug('文件路径：%s', file_path)
    if auto_open_file and file_path is not None:
        open_file_with_system_app(file_path)

    return file_path


def _test_df_2_table():
    df = pd.DataFrame({'a': list(range(10, 20)), 'b': list(range(10, 0, -1))})
    # 生成 docx 文件
    document = docx.Document()
    # 设置默认字体
    document.styles['Normal'].font.name = '微软雅黑'
    document.styles['Normal']._element.rPr.rFonts.set(docx.oxml.ns.qn('w:eastAsia'), '微软雅黑')
    # 创建自定义段落样式(第一个参数为样式名, 第二个参数为样式类型, 1为段落样式, 2为字符样式, 3为表格样式)
    UserStyle1 = document.styles.add_style('UserStyle1', 1)
    # 设置字体尺寸
    UserStyle1.font.size = docx.shared.Pt(40)
    # 设置字体颜色
    UserStyle1.font.color.rgb = docx.shared.RGBColor(0xff, 0xde, 0x00)
    # 居中文本
    UserStyle1.paragraph_format.alignment = docx.enum.text.WD_ALIGN_PARAGRAPH.CENTER
    # 设置中文字体
    UserStyle1.font.name = '微软雅黑'
    UserStyle1._element.rPr.rFonts.set(docx.oxml.ns.qn('w:eastAsia'), '微软雅黑')

    # 文件内容
    document.add_heading('测试使用', 0).alignment = docx.enum.text.WD_ALIGN_PARAGRAPH.CENTER
    document.add_paragraph('测试使用')

    # df_2_table(document, df, mark_top_n_on_cols=df.columns)
    # file_path = os.path.abspath(os.path.join(os.path.curdir, 'test.docx'))
    file_path = "/home/mg/github/test.docx"
    file_path = document.save(file_path)
    open_file_with_system_app(file_path)


if __name__ == "__main__":
    # _test_analysis_rewards_with_md()
    _test_df_2_table()
