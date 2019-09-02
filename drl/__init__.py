#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 2019/7/1 21:10
@File    : __init__.py.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
from ibats_utils.mess import is_windows_os
DATA_FOLDER_PATH = r'D:\WSPych\IBATSCommon\ibats_common\example\data' if is_windows_os() else r'/home/mg/github/IBATS_Common/ibats_common/example/data'