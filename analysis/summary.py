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
import itertools
import docx
import pandas as pd
from docx.shared import Pt, Inches
from ibats_common.analysis.plot import clean_cache
from ibats_common.analysis.summary import df_2_table, stats_df_2_docx_table
from ibats_common.backend.mess import get_report_folder_path
from ibats_utils.mess import datetime_2_str, date_2_str, str_2_date, open_file_with_system_app

logger = logging.getLogger(__name__)

STR_FORMAT_DATETIME_4_FILE_NAME = '%Y-%m-%d %H_%M_%S'
FORMAT_2_PERCENT = lambda x: f"{x * 100: .2f}%"
FORMAT_2_FLOAT2 = r"{0:.2f}"
FORMAT_2_FLOAT4 = r"{0:.4f}"
FORMAT_2_MONEY = r"{0:,.2f}"


def format_2_str(value, formator):
    """根据 formator 将对象格式化成 str"""
    if formator is None:
        text = str(value)
    elif isinstance(formator, str):
        text = str.format(formator, value)
    elif callable(formator):
        text = formator(value)
    else:
        raise ValueError('%s: %s 无效', value, formator)
    return text


def dic_2_table(doc, param_dic: dict, col_group_num=1, format_key=None, format_dic: (dict, None)=None, col_width=[1.75, 4.25]):
    # Highlight all cells limegreen (RGB 32CD32) if cell contains text "0.5"
    from docx.oxml.ns import nsdecls
    from docx.oxml import parse_xml
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    data_count = len(param_dic)
    if data_count == 0:
        return
    col_num_max = col_group_num * 2
    row_num_max = (data_count // col_group_num + 1) if data_count % col_group_num == 0 else \
        (data_count // col_group_num + 2)
    t = doc.add_table(row_num_max, col_num_max)

    # write head
    # col_name_list = list(sub_df.columns)
    for j in range(col_num_max):
        paragraph = t.cell(0, j).paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if j % 2 == 0:
            paragraph.add_run('key').bold = True
        else:
            paragraph.add_run('value').bold = True

    # write head bg color
    for j in range(col_num_max):
        # t.cell(0, j).text = df.columns[j]
        t.cell(0, j)._tc.get_or_add_tcPr().append(
            parse_xml(r'<w:shd {} w:fill="00A2E8"/>'.format(nsdecls('w'))))

    # format table style to be a grid
    t.style = 'TableGrid'

    # write key value
    for num, (key, value) in enumerate(param_dic.items()):
        row_num, col_num = num // col_group_num + 1, num % col_group_num * 2

        paragraph = t.cell(row_num, col_num).paragraphs[0]
        # populate the table with the dataframe
        text = format_2_str(key, format_key)

        paragraph.add_run(text).bold = True
        paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT

        if format_dic is not None and key in format_dic:
            format_cell = format_dic[key]
        else:
            format_cell = None

        text = format_2_str(value, format_cell)
        paragraph = t.cell(row_num, col_num + 1).paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        try:
            style = paragraph.add_run(text)
        except TypeError as exp:
            logger.exception('param_dic[%s] = %s', key, text)
            raise exp from exp

    # set alternative color
    for i in range(1, row_num_max):
        for j in range(col_num_max):
            if i % 2 == 0:
                t.cell(i, j)._tc.get_or_add_tcPr().append(
                    parse_xml(r'<w:shd {} w:fill="A3D9EA"/>'.format(nsdecls('w'))))

    # 如果出现多列，则将列的宽度 / 列数
    widths = (Inches(col_width[0] / col_group_num), Inches(col_width[1] / col_group_num))
    width_in_row = itertools.chain.from_iterable(itertools.repeat(widths, col_group_num))
    t.autofit = False
    # set col width
    # https://stackoverflow.com/questions/43051462/python-docx-how-to-set-cell-width-in-tables
    # It's not work
    # for row in t.rows:
    #     for cell, width in zip(row.cells, width_in_row):
    #         cell.width = width
    # it's work
    for idx, width in enumerate(width_in_row):
        t.columns[idx].width = width


def _test_dic_2_table():

    from docx.oxml.ns import nsdecls
    from docx.oxml import parse_xml
    from docx.enum.text import WD_ALIGN_PARAGRAPH

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
    UserStyle1.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    # 设置中文字体
    UserStyle1.font.name = '微软雅黑'
    UserStyle1._element.rPr.rFonts.set(docx.oxml.ns.qn('w:eastAsia'), '微软雅黑')

    # 文件内容
    heading_count, sub_heading_count = 0, 0
    document.add_heading('测试使用', 0).alignment = WD_ALIGN_PARAGRAPH.CENTER
    document.add_paragraph('')
    document.add_paragraph('')

    dic = {
        'from_date': str_2_date('2018-08-09'),
        'name': 'name test',
        'num': 12345,
        'float': 23456.789,
        'list': [1, 2, 3, 4],
        'long text': """bla bla bla（唧唧歪歪） 在英语中是“等等，之类的”的意思。
        当别人知道你要表达的意思时，用blablabla会比较方便，可要注意用的场合与人。在某种场合也形容某些人比较八卦。"""
    }
    file_path = 'test.docx'
    dic_2_table(document, dic, col_group_num=2, )
    document.save(file_path)
    open_file_with_system_app(file_path)
    return file_path


def summary_analysis_result_dic_2_docx(round_results_dic: dict, title_header,
                                       enable_clean_cache=False, doc_file_path=None):
    """对各个 round 分析结果进行汇总生产 docx 文件"""
    # 生成 docx 文档将所需变量
    heading_title = f'Rewards 汇总分析报告[{title_header}] '

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

    heading_count += 1
    sub_heading_count = 0
    key = 'available_episode_model_path_dic'
    document.add_heading(f'{heading_count}、各轮次训练有效的 episode 模型路径', 1)
    for num, (round_n, result_dic) in enumerate(round_results_dic.items()):
        analysis_result_dic = result_dic['analysis_result_dic']
        if key in analysis_result_dic:
            sub_heading_count += 1
            document.add_heading(f'{heading_count}.{sub_heading_count}、第 {round_n} 轮训练 ', 2)
            dic_2_table(document, analysis_result_dic[key], col_group_num=1)

    heading_count += 1
    document.add_heading(f'{heading_count}、各轮次训练 episode 与 value 变化趋势', 1)
    for num, (round_n, result_dic) in enumerate(round_results_dic.items()):
        document.add_heading(f'{heading_count}.{num}、Round {round_n} ', 2)
        sub_heading_count = 0
        key = 'param_dic'
        if key in result_dic:
            # 将测试运行参数输出到 word
            param_dic = result_dic[key]
            if param_dic is not None and len(param_dic) > 0:
                sub_heading_count += 1
                document.add_heading(f'{heading_count}.{num}.{sub_heading_count}、相关参数 ', 3)
                document.add_paragraph("""\t包括训练、模型路径等相关初始化参数""")
                format_dic = {}
                dic_2_table(document, param_dic, col_group_num=1, format_dic=format_dic)

        key, key1 = 'analysis_result_dic', 'episode_in_sample_value_plot'
        if key in result_dic and key1 in result_dic[key]:
            sub_heading_count += 1
            document.add_heading(f'{heading_count}.{num}.{sub_heading_count}、趋势变化 ', 3)
            file_path = result_dic[key][key1]
            document.add_picture(file_path)  # , width=docx.shared.Inches(1.25)
            document.add_paragraph('')

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


def summary_rewards_2_docx(param_dic, analysis_result_dic, title_header, enable_clean_cache=False,
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

    # 将测试运行参数输出到 word
    if param_dic is not None and len(param_dic) > 0:
        heading_count += 1
        document.add_heading(f'{heading_count}、相关参数', 1)
        document.add_paragraph("""\t包括训练、模型路径等相关初始化参数""")
        format_dic = {}
        dic_2_table(document, param_dic, col_group_num=1, format_dic=format_dic)

    # 随 Episode 增长，样本内 value 与 样本外 5日、10日、20日、60日 value 趋势对比
    # 目的是查看 样本内表现与样本外表现直接是否存在一定的相关性
    heading_count += 1
    document.add_heading(f'{heading_count}、样本内与样本外 5日、10日、20日、60日 Episode-value 趋势对比', 1)
    document.add_paragraph("""本节展示随 Episode 增长，样本内 value 与 样本外 5日、10日、20日、60日 value 趋势对比，
    目的是查看 样本内表现与样本外表现直接是否存在一定的相关性""")
    key = 'episode_value_compare'
    if key in analysis_result_dic:
        for n_days, file_path in analysis_result_dic[key].items():
            sub_heading_count += 1
            document.add_heading(f'{heading_count}、{sub_heading_count} 样本内与样本外{n_days}日对比走势', 1)
            document.add_picture(file_path)  # , width=docx.shared.Inches(1.25)
            document.add_paragraph('')

    # 样本内与样本外 value 相关性

    # 展示训练曲线 随 Episode 增长，value 结果变化曲线
    heading_count += 1
    document.add_heading(f'{heading_count}、展示训练曲线', 1)
    key = 'episode_in_sample_value_df'
    if key in analysis_result_dic:
        document.add_heading(f'{heading_count}.{sub_heading_count}、随 Episode 增长，value 结果变化曲线（样本内）'
                             f'{in_sample_date_line_str}', 2)
        document.add_picture(analysis_result_dic['episode_in_sample_value_plot'])
        document.add_paragraph('')

    key = 'episode_value_df'
    if key in analysis_result_dic:
        sub_heading_count += 1
        document.add_heading(f'{heading_count}.{sub_heading_count}、随 Episode 增长，value 结果变化曲线', 2)
        document.add_picture(analysis_result_dic['episode_value_plot'])  # , width=docx.shared.Inches(1.25)
        document.add_paragraph('详细数据')
        data_df = analysis_result_dic[key]

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
    for num, file_path in enumerate(analysis_result_dic['value_plot_list'], start=1):
        document.add_paragraph(f'{heading_count}.{sub_heading_count}.{num}')
        document.add_picture(file_path)
        document.add_paragraph('')
    data_df = analysis_result_dic['episode_reward_df'][available_reward_col_name_list]
    format_by_col = {_: FORMAT_2_FLOAT4 for _ in data_df.columns if _ in float_col_name_set}
    df_2_table(document, data_df, format_by_col=format_by_col, max_col_count=6,
               mark_top_n=5, mark_top_n_on_cols=data_df.columns)
    document.add_paragraph('')
    key = 'value_plot_dic'
    if key in analysis_result_dic:
        sub_heading_count += 1
        document.add_heading(f'{heading_count}.{sub_heading_count}、详细数据', 2)
        # 分别展示
        for num, (episode, file_path) in enumerate(analysis_result_dic[key].items(), start=2):
            document.add_heading(f'{heading_count}.{sub_heading_count}.{num}、episode={episode}', 2)
            document.add_picture(file_path)
            document.add_paragraph('')
            data_df = analysis_result_dic['episode_reward_dic'][episode][available_reward_col_name_list]
            format_by_col = {_: FORMAT_2_FLOAT4 for _ in data_df.columns if _ in float_col_name_set}
            df_2_table(document, data_df, format_by_col=format_by_col, max_col_count=6)
            document.add_paragraph('')

    heading_count += 1
    document.add_page_break()

    document.add_heading(f'{heading_count}、样本外数据绩效统计数据（Porformance stat）', 1)
    key = 'stats_df'
    if key in analysis_result_dic:
        stats_df = analysis_result_dic[key].T
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


if __name__ == '__main__':
    _test_dic_2_table()
