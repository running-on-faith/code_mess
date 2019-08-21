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

from ibats_common.example.drl.d3qn_replay_2019_08_07.agent.main import get_agent, MODEL_NAME


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
    agent = get_agent(input_shape=batch_factors.shape)
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
        title = f'{MODEL_NAME}_in_sample_{dt_str}' if key is None else f'ddqn_lstm2_in_sample_episode_{key}_{dt_str}'
        from ibats_common.analysis.plot import plot_twin
        plot_twin(value_df, md_df["close"], name=title, folder_path='images')

    logger.debug("模型：%s，样本内测试[batch=%s]完成", model_path, batch)
    return reward_df


def _test_load_predict(model_folder='model', target_round_n=1, show_plot_together=True):
    import pandas as pd
    import os
    from ibats_common.example.data import load_data
    from ibats_common.backend.factor import get_factor, transfer_2_batch
    from ibats_utils.mess import datetime_2_str
    from datetime import datetime
    from ibats_common.analysis.plot import plot_twin
    logger = logging.getLogger(__name__)
    # 建立相关数据
    n_step = 60  # 此处要与main 保持一致
    ohlcav_col_name_list = ["open", "high", "low", "close", "amount", "volume"]

    md_df = load_data('RB.csv').set_index('trade_date')[ohlcav_col_name_list]
    md_df.index = pd.DatetimeIndex(md_df.index)
    factors_df = get_factor(md_df, dropna=True)
    df_index, df_columns, data_arr_batch = transfer_2_batch(factors_df, n_step=n_step)
    data_factors, shape = data_arr_batch, data_arr_batch.shape
    # shape = [data_arr_batch.shape[0], 5, int(n_step / 5), data_arr_batch.shape[2]]
    # data_factors = np.transpose(data_arr_batch.reshape(shape), [0, 2, 3, 1])
    # print(data_arr_batch.shape, '->', shape, '->', data_factors.shape)
    md_df = md_df.loc[df_index, :]
    predict_result_dic, episode_model_path_dic, episode_list = {}, {}, []
    for file_name in os.listdir(model_folder):
        file_path = os.path.join(model_folder, file_name)
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
        episode = int(episode)
        episode_model_path_dic[episode] = file_path
        episode_list.append(episode)

    if len(episode_list) == 0:
        logger.info('target_round_n=%d 没有可加载的模型', target_round_n)
        return

    episode_reward_df_dic = {}
    episode_list.sort()
    episode_count = len(episode_list)
    for num, episode in enumerate(episode_list, start=1):
        file_path = episode_model_path_dic[episode]
        logger.debug('%2d/%2d ) %4d -> %s', num, episode_count, episode, file_path)
        reward_df = load_predict(md_df, data_factors, tail_n=0, model_path=file_path, key=episode,
                                 show_plot=False)
        # reward_df.to_csv(f'reward_{round_n}_{file_name_no_extension}.csv')
        predict_result_dic[episode] = reward_df.iloc[-1, :]
        episode_reward_df_dic[episode] = reward_df

    import matplotlib.pyplot as plt
    fig, ax = plt.figure(figsize=(8, 12)), None  #
    if show_plot_together:
        ax = fig.add_subplot(211)
        # 前两个以及最后一个输出，其他的能整除才输出
        mod = int(episode_count / 5)
        value_df = pd.DataFrame({f'{episode}_v': episode_reward_df_dic[episode]['value']
                                 for num, episode in enumerate(episode_list)
                                 if episode_reward_df_dic[episode].shape[0] > 0 and (
                                              num % mod == 0 or num in (1, episode_count - 1))})
        value_fee0_df = pd.DataFrame({f'{episode}_0': episode_reward_df_dic[episode]['value_fee0']
                                      for num, episode in enumerate(episode_list)
                                      if episode_reward_df_dic[episode].shape[0] > 0 and (
                                              num % mod == 0 or num in (1, episode_count - 1))})

        dt_str = datetime_2_str(datetime.now(), '%Y-%m-%d %H%M%S')
        title = f'{MODEL_NAME}_validation_r{target_round_n}_{dt_str}'
        plot_twin([value_df, value_fee0_df], md_df['close'], ax=ax, name=title,
                  enable_save_plot=False, enable_show_plot=False, do_clr=False)

    # 展示训练曲线
    if len(predict_result_dic) > 0:
        if ax is not None:
            ax = fig.add_subplot(212)
        predict_result_df = pd.DataFrame(predict_result_dic).T.sort_index()
        dt_str = datetime_2_str(datetime.now(), '%Y-%m-%d %H%M%S')
        title = f'{MODEL_NAME}_summary_r{target_round_n}_{dt_str}_trend'
        plot_twin(predict_result_df[['value', 'value_fee0']], predict_result_df['action_count'],
                  ax=ax, name=title, y_scales_log=[False, True], folder_path='images')


if __name__ == "__main__":
    _test_load_predict(target_round_n=25)
    # for _ in range(0, 25):
    #     _test_load_predict(target_round_n=_)
