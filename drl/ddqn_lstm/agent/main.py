#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 2019/6/28 17:19
@File    : main.py
@contact : mmmaaaggg@163.com
@desc    :
需要安装相关包
pip install graphs dm-sonnet==1.19
https://github.com/deepmind/sonnet
dm-sonnet==1.19 对应 tensorflow==1.5.1
"""
import logging

import tensorflow as tf

from ibats_common.example.drl.ddqn_lstm.agent.framework import Framework


class Agent(object):
    def __init__(self, input_shape=None, **kwargs):
        tf.reset_default_graph()
        if input_shape is not None:
            input_shape = list(input_shape)
            input_shape[0] = None
            self.agent = Framework(input_shape=input_shape, **kwargs)
        else:
            self.agent = Framework(**kwargs)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

    def choose_action_deterministic(self, inputs):
        return self.agent.get_deterministic_policy(inputs)

    def choose_action_deterministic_batch(self, inputs):
        return self.agent.get_deterministic_policy_batch(inputs)

    def choose_action_stochastic(self, inputs):
        return self.agent.get_stochastic_policy(inputs)

    def update_cache(self, state, action, reward, next_state, done):
        self.agent.update_cache(state, action, reward, next_state, done)

    def update_eval(self):
        self.agent.update_value_net()

    def update_target(self):
        self.agent.update_target_net()

    def save_model(self, path="model/weights.h5"):
        # return self.saver.save(self.sess, path)
        self.agent.model.save_weights(path)

    def restore_model(self, path="model/weights.h5"):
        # self.saver.restore(self.sess, path)
        self.agent.model.load_weights(path)

    def close(self):
        # self.sess.close()
        pass


def _test_agent():
    import pandas as pd
    # 建立相关数据
    n_step = 60
    ohlcav_col_name_list = ["open", "high", "low", "close", "amount", "volume"]
    from ibats_common.example.data import load_data
    md_df = load_data('RB.csv').set_index('trade_date')[ohlcav_col_name_list]
    md_df.index = pd.DatetimeIndex(md_df.index)
    from ibats_common.backend.factor import get_factor, transfer_2_batch
    factors_df = get_factor(md_df, dropna=True)
    df_index, df_columns, data_arr_batch = transfer_2_batch(factors_df, n_step=n_step)
    md_df = md_df.loc[df_index, :]

    from ibats_common.backend.rl.emulator.account import Account
    env = Account(md_df, data_arr_batch)
    agent = Agent()
    # fill cache
    for episode in range(2):
        state = env.reset()
        while True:
            action = agent.choose_action_stochastic(state)
            next_state, reward, done = env.step(action)
            agent.update_cache(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
    print(len(agent.agent.cache))
    reward_df = env.plot_data()
    reward_df.to_csv('reward_df.csv')


def train(md_df, batch_factors, round_n=0, num_episodes=200, n_episode_pre_record=20):
    import pandas as pd
    from ibats_common.backend.rl.emulator.account import Account
    logger = logging.getLogger(__name__)
    env = Account(md_df, data_factors=batch_factors)
    agent = Agent(input_shape=batch_factors.shape, action_size=env.action_size, gamma=0.3, memory_size=25000)
    # num_episodes, n_episode_pre_record = 200, 20

    target_step_size = 512
    train_step_size = 64

    episodes_train = []
    global_step = 0
    for episode in range(num_episodes):
        state = env.reset()
        episode_step = 0
        while True:
            global_step += 1
            episode_step += 1

            action = agent.choose_action_stochastic(state)
            next_state, reward, done = env.step(action)
            agent.update_cache(state, action, reward, next_state, done)
            state = next_state

            if global_step % target_step_size == 0:
                agent.update_target()
                # print('global_step=%d, episode_step=%d, agent.update_target()' % (global_step, episode_step))

            if (global_step >= train_step_size and episode_step % train_step_size == 0) or done:
                agent.update_eval()
                # print('global_step=%d, episode_step=%d, agent.update_eval()' % (global_step, episode_step))

                if done:
                    # print("episode=%d, data_observation.shape[0]=%d, env.A.total_value=%f" % (
                    #     episode, env.A.data_observation.shape[0], env.A.total_value))
                    if episode % n_episode_pre_record == 0 or episode == num_episodes - 1:
                        if round_n is None:
                            logger.debug("episode=%d, data_observation.shape[0]=%d, env.A.total_value=%f",
                                         episode, env.A.data_observation.shape[0], env.A.total_value)
                        else:
                            logger.debug("round=%d, episode=%3d, env.A.total_value=%f",
                                         round_n, episode, env.A.total_value)
                        episodes_train.append(env.plot_data())
                    break

    # 输出图表
    import matplotlib.pyplot as plt
    reward_df = env.plot_data()
    reward_df.iloc[:, 0].plot()  # figsize=(16, 6)
    import datetime
    from ibats_utils.mess import datetime_2_str
    dt_str = datetime_2_str(datetime.datetime.now(), '%Y-%m-%d %H%M%S')
    title = f'ddqn_lstm_round_n_{round_n}_{dt_str}'
    plt.suptitle(title)
    plt.show()
    value_df = pd.DataFrame({num: df['value']
                             for num, df in enumerate(episodes_train, start=1)
                             if df.shape[0] > 0})
    from ibats_common.analysis.plot import plot_twin
    plot_twin(value_df, md_df["close"], name=title)
    # if reward_df.iloc[-1, 0] > reward_df.iloc[0, 0]:
    path = f"model/weights_{round_n}.h5"
    agent.save_model(path=path)
    logger.debug('model save to path: %s', path)
    agent.close()
    return reward_df


def _test_agent2():
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

    success_count, success_max_count, round_n = 0, 10, 0
    while True:
        round_n += 1
        # 执行训练
        df = train(md_df, batch_factors, round_n=round_n,
                   num_episodes=200, n_episode_pre_record=20)
        logger.debug(df.iloc[-1, :])
        if df["value"].iloc[-1] > df["value"].iloc[0]:
            success_count += 1
            logger.debug('is win %d/%d' % (success_count, success_max_count))
            if success_count >= success_max_count:
                break


def load_predict(md_df, data_factors, tail_n=1, show_plot=True, model_path="model/weights_1.h5", batch=True):
    from ibats_common.backend.rl.emulator.account import Account
    logger = logging.getLogger(__name__)
    if tail_n is not None and tail_n > 0:
        states = data_factors[-tail_n:]
        md_df = md_df.iloc[-tail_n:]
    else:
        states = data_factors

    env = Account(md_df, data_factors=data_factors)
    agent = Agent(input_shape=data_factors.shape)
    agent.restore_model(path=model_path)
    logger.debug("加载模型：%s 完成", model_path)

    if batch:
        # 批量作业
        actions = agent.choose_action_deterministic_batch(states)
        for num, action in enumerate(actions, start=1):
            next_state, reward, done = env.step(action)
            if done:
                logger.debug('执行循环 %d 次', num)
                break
    else:
        # 单步执行
        done, state, num = False, env.reset(), 0
        while not done:
            num += 1
            action = agent.choose_action_deterministic(state)
            state, reward, done = env.step(action)
            if done:
                logger.debug('执行循环 %d 次', num)
                break

    reward_df = env.plot_data()
    if show_plot:
        import matplotlib.pyplot as plt
        value_s = reward_df.iloc[:, 0]
        from ibats_utils.mess import datetime_2_str
        from datetime import datetime
        dt_str = datetime_2_str(datetime.now(), '%Y-%m-%d %H%M%S')
        title = f'ddqn_lstm_predict_{dt_str}'
        from ibats_common.analysis.plot import plot_twin
        plot_twin(value_s, md_df["close"], name=title)

    return reward_df


def _test_load_predict():
    import pandas as pd
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
    model_path = f"model/weights_25.h5"
    reward_df = load_predict(md_df, data_factors, tail_n=0, model_path=model_path)
    reward_df.to_csv('reward_df.csv')


if __name__ == '__main__':
    # _test_agent()
    # _test_agent2()
    _test_load_predict()
