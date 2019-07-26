#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 19-7-25 下午1:15
@File    : validate.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
import logging
from ibats_common.example.drl.ddqn_lstm2.agent.main import Agent


def load_predict(md_df, batch_factors, tail_n=1, show_plot=True, model_path="model/weights_1.h5", batch=False,
                 key=None):
    """加载模型，进行样本内训练"""
    import numpy as np
    from ibats_common.backend.rl.emulator.account import Account
    logger = logging.getLogger(__name__)
    if tail_n is not None and tail_n > 0:
        states = batch_factors[-tail_n:]
        md_df = md_df.iloc[-tail_n:]
    else:
        states = batch_factors

    env = Account(md_df, data_factors=batch_factors, state_with_flag=True)
    agent = Agent(input_shape=batch_factors.shape, action_size=3, dueling=True,
                  gamma=0.3, batch_size=512, memory_size=100000)
    agent.restore_model(path=model_path)
    logger.debug("模型：%s 加载完成，样本内测试[batch=%s]开始", model_path, batch)

    if batch:
        # 批量作业，批量执行速度快，但还四可能与实际清空所有偏差
        # 伪 flag
        flags = np.zeros((states.shape[0], 1))
        actions = agent.choose_action_deterministic_batch((states, flags))
        for num, action in enumerate(actions, start=1):
            next_state, reward, done = env.step(action)
            if done:
                logger.debug('执行循环 %d / %d 次', num, md_df.shape[0])
                break
    else:
        # 单步执行
        done, state, num = False, env.reset(), 0
        while not done:
            num += 1
            action = agent.choose_action_deterministic(state)
            state, reward, done = env.step(action)
            if done:
                logger.debug('执行循环 %d / %d 次', num, md_df.shape[0])
                break

    reward_df = env.plot_data()
    if show_plot:
        value_df = reward_df[['value', 'value_fee0']] / env.A.init_cash
        from ibats_utils.mess import datetime_2_str
        from datetime import datetime
        dt_str = datetime_2_str(datetime.now(), '%Y-%m-%d %H%M%S')
        title = f'ddqn_lstm2_in_sample_{dt_str}' if key is None else f'ddqn_lstm2_in_sample_episode_{key}_{dt_str}'
        from ibats_common.analysis.plot import plot_twin
        plot_twin(value_df, md_df["close"], name=title)

    logger.debug("模型：%s，样本内测试[batch=%s]完成", model_path, batch)
    return reward_df


def _test_load_predict(model_folder='model', target_round_n=1):
    import pandas as pd
    import os
    import matplotlib.pyplot as plt
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
    data_factors, shape = data_arr_batch, data_arr_batch.shape
    # shape = [data_arr_batch.shape[0], 5, int(n_step / 5), data_arr_batch.shape[2]]
    # data_factors = np.transpose(data_arr_batch.reshape(shape), [0, 2, 3, 1])
    # print(data_arr_batch.shape, '->', shape, '->', data_factors.shape)
    md_df = md_df.loc[df_index, :]
    predict_result_dic = {}
    for file_name in os.listdir(model_folder):
        file_path = os.path.join(model_folder, file_name)
        logger.debug(file_path)
        if os.path.isdir(file_path):
            continue
        file_name_no_extension, extension = os.path.splitext(file_name)
        if extension != '.h5':
            continue
        # _, file_name_no_extension = os.path.split(file_name)
        # model_path = f"model/weights_1.h5"
        _, round_n, episode = file_name_no_extension.split('_')
        if int(round_n) != target_round_n:
            continue
        reward_df = load_predict(md_df, data_factors, tail_n=0, model_path=file_path, key=episode)
        reward_df.to_csv(f'reward_{round_n}_{file_name_no_extension}.csv')
        predict_result_dic[episode] = reward_df.iloc[-1, :]

    # 展示训练曲线
    if len(predict_result_dic) > 0:
        predict_result_df = pd.DataFrame(predict_result_dic).T
        predict_result_df[['value', 'value_fee0', 'fee_tot']].plot()
        from ibats_utils.mess import datetime_2_str
        from datetime import datetime
        dt_str = datetime_2_str(datetime.now(), '%Y-%m-%d %H%M%S')
        title = f'ddqn_lstm2_summary_r{target_round_n}_{dt_str}_trend'
        plt.suptitle(title)
        file_path = f'{title}.png'
        from ibats_common.analysis.plot import plot_or_show
        plot_or_show(file_name=file_path)


if __name__ == "__main__":
    _test_load_predict()