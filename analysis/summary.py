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
import pandas as pd
from docx.shared import Pt
from ibats_common.analysis.plot import clean_cache
from ibats_common.analysis.summary import df_2_table, stats_df_2_docx_table
from ibats_common.backend.mess import get_report_folder_path
from ibats_utils.mess import datetime_2_str, date_2_str

logger = logging.getLogger(__name__)

STR_FORMAT_DATETIME_4_FILE_NAME = '%Y-%m-%d %H_%M_%S'
FORMAT_2_PERCENT = lambda x: f"{x * 100: .2f}%"
FORMAT_2_FLOAT2 = r"{0:.2f}"
FORMAT_2_FLOAT4 = r"{0:.4f}"
FORMAT_2_MONEY = r"{0:,.2f}"


def summary_rewards_2_docx(result_dic, title_header, enable_clean_cache=False,
                           doc_file_path=None, in_sample_date_line=None):
    """创建 绩效分析结果 word 文档"""
    # 整理参数
    in_sample_date_line = None if in_sample_date_line is None else pd.to_datetime(in_sample_date_line)
    in_sample_date_line_str = date_2_str(in_sample_date_line) if in_sample_date_line is not None else ""
    int_col_name_set = {'action', 'action_count'}
    money_col_name_set = {'value', 'cash', 'fee_tot', 'value_fee0'}
    float_col_name_set = {'value', 'cash', 'close', 'fee_tot', 'value_fee0'}
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
    heading_count, sub_heading_count = 0, 0
    document.add_heading(heading_title, 0).alignment = docx.enum.text.WD_ALIGN_PARAGRAPH.CENTER
    document.add_paragraph('')
    document.add_paragraph('')

    # 随 Episode 增长，样本内 value 与 样本外 5日、10日、20日、60日 value 趋势对比
    # 目的是查看 样本内表现与样本外表现直接是否存在一定的相关性
    heading_count += 1
    document.add_heading(f'{heading_count}、样本内与样本外 5日、10日、20日、60日 Episode-value 趋势对比', 1)
    document.add_paragraph("""本节展示随 Episode 增长，样本内 value 与 样本外 5日、10日、20日、60日 value 趋势对比，
    目的是查看 样本内表现与样本外表现直接是否存在一定的相关性""")
    key = 'episode_value_compare'
    if key in result_dic:
        for n_days, file_path in result_dic[key].items():
            sub_heading_count += 1
            document.add_heading(f'{heading_count}、{sub_heading_count} {n_days}日对比走势', 1)
            document.add_picture(file_path)  # , width=docx.shared.Inches(1.25)
            document.add_paragraph('')

    # 样本内与样本外 value 相关性

    # 展示训练曲线 随 Episode 增长，value 结果变化曲线
    heading_count += 1
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

        format_by_col = {_: FORMAT_2_FLOAT4 for _ in data_df.columns if _ in float_col_name_set}
        format_by_col.update({_: FORMAT_2_PERCENT for _ in data_df.columns if _.find('rr') > 0 or _.find('cagr') > 0})
        df_2_table(document, data_df, format_by_col=format_by_col, max_col_count=6,
                   mark_top_n=5, mark_top_n_on_cols=data_df.columns)
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
    format_by_col = {_: FORMAT_2_FLOAT4 for _ in data_df.columns if _ in float_col_name_set}
    df_2_table(document, data_df, format_by_col=format_by_col, max_col_count=6,
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
            format_by_col = {_: FORMAT_2_FLOAT4 for _ in data_df.columns if _ in float_col_name_set}
            df_2_table(document, data_df, format_by_col=format_by_col, max_col_count=6)
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
        file_name = f"{title_header}_" \
            f"{datetime_2_str(datetime.datetime.now(), STR_FORMAT_DATETIME_4_FILE_NAME)}.docx"
        file_path = os.path.join(folder_path, file_name)
    else:
        file_path = doc_file_path

    document.save(file_path)
    if enable_clean_cache:
        clean_cache()

    return file_path
