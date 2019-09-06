#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 19-9-6 上午7:23
@File    : __init__.py.py
@contact : mmmaaaggg@163.com
@desc    : 
"""


def set_module_root_path():
    import os
    import re
    from ibats_utils.mess import get_folder_path
    module_root_path = get_folder_path(re.compile(r'^code_mess'), create_if_not_found=False)  # 'ibats_common'
    import ibats_common
    ibats_common.module_root_path = module_root_path
    ibats_common.root_parent_path = os.path.abspath(os.path.join(module_root_path, os.path.pardir)) \
        if module_root_path is not None else None


set_module_root_path()

if __name__ == "__main__":
    pass
