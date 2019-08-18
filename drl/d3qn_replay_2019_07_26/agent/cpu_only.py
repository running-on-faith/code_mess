#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 2019/7/28 21:02
@File    : test.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import logging
from ibats_common.example.drl.d3qn_replay_2019_07_26.agent.main import train
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from tensorflow.python.client import device_lib


def show_device():
    return device_lib.list_local_devices()


def _test_agent2():
    """测试模型训练过程"""
    import pandas as pd
    logger = logging.getLogger(__name__)
    # 建立相关数据
    n_step = 250
    ohlcav_col_name_list = ["open", "high", "low", "close", "amount", "volume"]
    from ibats_common.example.data import load_data
    md_df = load_data('RB.csv').set_index('trade_date')[ohlcav_col_name_list]
    md_df.index = pd.DatetimeIndex(md_df.index)
    from ibats_common.backend.factor import get_factor, transfer_2_batch
    factors_df = get_factor(md_df, dropna=True)
    df_index, df_columns, data_arr_batch = transfer_2_batch(factors_df, n_step=n_step)
    batch_factors = data_arr_batch
    # shape = [data_arr_batch.shape[0], 5, int(n_step / 5), data_arr_batch.shape[2]]
    # batch_factors = np.transpose(data_arr_batch.reshape(shape), [0, 2, 3, 1])
    # print(data_arr_batch.shape, '->', shape, '->', batch_factors.shape)
    md_df = md_df.loc[df_index, :]

    # success_count, success_max_count, round_n = 0, 10, 0
    round_max, round_n, increase = 40, 0, 100
    for round_n in range(round_n, round_max):
        round_n += 1
        # 执行训练
        num_episodes = 200 + round_n * increase
        df, path = train(md_df, batch_factors, round_n=round_n,
                         num_episodes=num_episodes,
                         n_episode_pre_record=int(num_episodes / 6))
        logger.debug('round_n=%d, final status:\n%s', round_n, df.iloc[-1, :])


if __name__ == '__main__':
    devices = show_device()
    print(type(devices), len(devices))
    for num, d in enumerate(devices):
        print(num, '\n', d)
    # KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))
    KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))
    _test_agent2()
