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
from docx.shared import Pt
from ibats_common.analysis import get_report_folder_path
from ibats_common.analysis.plot import clean_cache
from ibats_common.analysis.plot import plot_twin
from ibats_common.analysis.summary import df_2_table, stats_df_2_docx_table
from ibats_utils.mess import date_2_str, datetime_2_str, open_file_with_system_app

from drl import DATA_FOLDER_PATH

logger = logging.getLogger(__name__)
logger.debug('import %s', ffn)
STR_FORMAT_DATETIME_4_FILE_NAME = '%Y-%m-%d %H_%M_%S'
FORMAT_2_PERCENT = lambda x: f"{x * 100: .2f}%"
FORMAT_2_FLOAT2 = r"{0:.2f}"
FORMAT_2_FLOAT4 = r"{0:.4f}"


def analysis_rewards_with_md(episode_reward_df_dic, title_header, md_df, enable_show_plot=False,
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

    # 展示训练曲线
    episode_list.sort()
    predict_result_dic = {_: val.iloc[-1, :] for _, val in episode_reward_df_dic.items()}
    predict_result_df = pd.DataFrame(predict_result_dic).T.sort_index()
    title = f'{title_header}_episode_trend_summary'
    file_path = plot_twin(predict_result_df[['value', 'value_fee0']], predict_result_df['action_count'],
                          folder_path=cache_folder_path,
                          name=title, y_scales_log=[False, True], **enable_kwargs)
    # logger.debug("predict_result_df=\n%s", predict_result_df)
    result_dic['episode_trend_summary_plot'] = file_path
    result_dic['episode_trend_summary_df'] = predict_result_df

    # value，value_fee0 走势图
    # 每一个 reward 一张图
    if show_plot_141:
        value_plot_dic, episode_reward_dic = {}, {}
        for num, episode in enumerate(episode_list):
            if episode_reward_df_dic[episode].shape[0] == 0:
                continue
            value_df = pd.DataFrame({f'{episode}_v': episode_reward_df_dic[episode]['value'],
                                     f'{episode}_0': episode_reward_df_dic[episode]['value_fee0']})

            # 例如：d3qn_in_sample_205-01-01_r0_2019-09-10
            title = f'{title_header}_value_{episode}'
            file_path = plot_twin([value_df[f'{episode}_v'], value_df[f'{episode}_0']], md_df['close'],
                                  name=title,
                                  folder_path=cache_folder_path,
                                  in_sample_date_line=in_sample_date_line, **enable_kwargs)
            value_plot_dic[episode] = file_path
            episode_reward_dic[episode] = episode_reward_df_dic[episode].iloc[-1:, :]

        result_dic['value_plot_dic'] = value_plot_dic
        result_dic['episode_reward_dic'] = episode_reward_dic

    # value，value_fee0 走势图
    # 合并展示图
    if episode_count > 10:
        mod = int(episode_count / 5)
    else:
        mod = 1

    value_df = pd.DataFrame({f'{episode}_v': episode_reward_df_dic[episode]['value']
                             for num, episode in enumerate(episode_list)
                             if episode_reward_df_dic[episode].shape[0] > 0 and (
                                     num % mod == 0 or num in (1, episode_count - 1))})
    value_fee0_df = pd.DataFrame({f'{episode}_0': episode_reward_df_dic[episode]['value_fee0']
                                  for num, episode in enumerate(episode_list)
                                  if episode_reward_df_dic[episode].shape[0] > 0 and (
                                          num % mod == 0 or num in (1, episode_count - 1))})

    # 例如：d3qn_in_sample_205-01-01_r0_2019-09-10
    title = f"{title_header}_value_tot"
    file_path = plot_twin([value_df, value_fee0_df], md_df['close'], name=title,
                          folder_path=cache_folder_path,
                          in_sample_date_line=in_sample_date_line, **enable_kwargs)
    result_dic['value_plot'] = file_path
    result_dic['episode_reward_df'] = pd.DataFrame(
        {episode: df.iloc[-1, :] for episode, df in episode_reward_df_dic.items()}
        ).T.sort_index()

    # in_sample_date_line 节点后绩效分析
    perfomance_dic = {}
    for num, (episode, reward_df) in enumerate(episode_reward_df_dic.items()):
        if in_sample_date_line is not None:
            reward_df = reward_df[reward_df.index >= in_sample_date_line]

        df = reward_df[['value', 'value_fee0']] / reward_df[['value', 'value_fee0']]
        df.rename(columns={'value': f'{episode}_value', 'value_fee0': f'{episode}_value_fee0'}, inplace=True)
        if num == 0:
            df['close'] = md_df.loc[reward_df.index, 'close']

        # 绩效分析
        stats_result = df.calc_stats()
        stats_result.set_riskfree_rate(risk_free)
        perfomance_dic.update({_: stats.stats for _, stats in stats_result.items()})

    result_dic['stats_df'] = pd.DataFrame(perfomance_dic).T

    return result_dic


def summary_4_rewards(result_dic, title_header, md_df, enable_clean_cache=False,
                      doc_file_path=None):
    """创建 绩效分析结果 word 文档"""
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
    heading_count = 1
    document.add_heading(f'{heading_count}、展示训练曲线', 1)
    # 增加图片（此处使用相对位置）
    document.add_picture(result_dic['episode_trend_summary_plot'])  # , width=docx.shared.Inches(1.25)
    document.add_paragraph('')
    available_reward_col_name_list = ['value', 'cash', 'fee_tot', 'value_fee0', 'action_count']
    key = 'episode_trend_summary_df'
    if key in result_dic:
        data_df = result_dic[key][available_reward_col_name_list]
        int_col_name_set = {'action', 'action_count'}
        format_by_col = {_: FORMAT_2_FLOAT2 for _ in data_df.columns if _ not in int_col_name_set}
        df_2_table(document, data_df, format_by_col=format_by_col, max_col_count=5)
        heading_count += 1
        # 添加分页符
        document.add_page_break()

    # 展示 Value Plot
    document.add_heading(f'{heading_count}、Reward Value 曲线走势', 1)
    # 汇总展示
    document.add_heading(f'{heading_count}.1、汇总走势', 2)
    file_path = result_dic['value_plot']
    document.add_picture(file_path)
    document.add_paragraph('')
    data_df = result_dic['episode_reward_df'][available_reward_col_name_list]
    format_by_col = {_: FORMAT_2_FLOAT2 for _ in data_df.columns if _ not in int_col_name_set}
    df_2_table(document, data_df, format_by_col=format_by_col, max_col_count=5)
    document.add_paragraph('')
    key = 'value_plot_dic'
    if key in result_dic:
        # 分别展示
        for num, (episode, file_path) in enumerate(result_dic[key].items(), start=2):
            document.add_heading(f'{heading_count}.{num}、episode={episode}', 2)
            document.add_picture(file_path)
            document.add_paragraph('')
            data_df = result_dic['episode_reward_dic'][episode][available_reward_col_name_list]
            format_by_col = {_: FORMAT_2_FLOAT2 for _ in data_df.columns if _ not in int_col_name_set}
            df_2_table(document, data_df, format_by_col=format_by_col, max_col_count=5)
            document.add_paragraph('')

    heading_count += 1
    document.add_page_break()

    document.add_heading(f'{heading_count}、样本外数据绩效统计数据（Porformance stat）', 1)
    stats_df = result_dic['stats_df']
    stats_df.drop(['start', 'rf'], axis=1)
    stats_df_2_docx_table(stats_df, document)
    heading_count += 1
    document.add_page_break()

    # 保存文件
    if doc_file_path is not None:
        if os.path.isdir(doc_file_path):
            folder_path, file_name = doc_file_path, ''
        else:
            folder_path, file_name = os.path.split(doc_file_path)
    else:
        folder_path, file_name = get_report_folder_path(), ''

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
    target_round_n = 0
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
    result_dic = analysis_rewards_with_md(episode_reward_df_dic, title_header, md_df,
                                          in_sample_date_line=in_sample_date_line, show_plot_141=True)
    file_path = summary_4_rewards(result_dic, title_header, md_df,
                                  doc_file_path=f'/home/mg/github/code_mess/output/reports/{title_header}.docx'
                                  )
    logger.debug('文件路径：%s', file_path)
    if auto_open_file and file_path is not None:
        open_file_with_system_app(file_path)

    return file_path


if __name__ == "__main__":
    _test_analysis_rewards_with_md()
