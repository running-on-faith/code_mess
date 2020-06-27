#! /usr/bin/env python3
"""
@author  : MG
@Time    : 2020/2/9 上午9:14
@File    : __init__.py.py
@contact : mmmaaaggg@163.com
@desc    : 
"""


def plot_modal_2_file(model, file_name, auto_open=True):
    from keras.utils import plot_model
    from ibats_utils.mess import open_file_with_system_app
    file_path = file_name
    plot_model(model, to_file=file_path, show_shapes=True)
    if auto_open:
        open_file_with_system_app(file_path)


if __name__ == "__main__":
    pass
