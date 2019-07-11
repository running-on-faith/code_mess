#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 19-7-10 上午9:35
@File    : main.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
import tensorflow as tf
from ibats_common.example.drl.d3qn2.agent.ddqn import DDQN


class Agent(object):
    def __init__(self, input_shape=None, action_size=3, **kwargs):
        tf.reset_default_graph()
        input_shape = list(input_shape)
        input_shape[0] = None
        self.agent = DDQN(action_dim=action_size, state_dim=input_shape, **kwargs)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        self.sess.graph.finalize()

    def choose_action_deterministic(self, state):
        return self.agent.agent.predict(state)[0]

    def choose_action_stochastic(self, state):
        return self.agent.policy_action(state)

    def update_cache(self, state, action, reward, next_state, done):
        self.agent.memorize(state, action, reward, done, next_state)

    def update_eval(self):
        self.agent.update_value_net(self.sess)

    def update_target(self):
        self.agent.update_target_net(self.sess)

    def save_model(self, path="model/ddqn.ckpt"):
        return self.agent.save_weights(path)

    def restore_model(self, path="model/ddqn.ckpt"):
        self.agent.load_weights(path)

    def close(self):
        self.sess.close()


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
            action = agent.choose_action_stochastic(state, 0)
            next_state, reward, done = env.step(action)
            agent.update_cache(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
    print(len(agent.agent.cache))
    reward_df = env.plot_data()
    reward_df.to_csv('reward_df.csv')


def train(md_df, batch_factors, round_n=None):
    import pandas as pd
    from ibats_common.backend.rl.emulator.account import Account
    env = Account(md_df, data_factors=batch_factors, expand_dims=False)
    agent = Agent(input_shape=batch_factors.shape)
    num_episodes = 500

    target_step_size = 128
    train_step_size = 32

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

            if episode_step % train_step_size == 0 or done:
                agent.update_eval()
                # print('global_step=%d, episode_step=%d, agent.update_eval()' % (global_step, episode_step))

                if done:
                    # print("episode=%d, data_observation.shape[0]=%d, env.A.total_value=%f" % (
                    #     episode, env.A.data_observation.shape[0], env.A.total_value))
                    if episode % 100 == 0 or episode == num_episodes - 1:
                        if round_n is None:
                            print("episode=%d, data_observation.shape[0]=%d, env.A.total_value=%f" % (
                                episode, env.A.data_observation.shape[0], env.A.total_value))
                        else:
                            print("round=%d, episode=%d, env.A.total_value=%f" % (
                                round_n, episode, env.A.total_value))
                        episodes_train.append(env.plot_data())
                    break

    import matplotlib.pyplot as plt
    reward_df = env.plot_data()
    reward_df.iloc[:, 0].plot(figsize=(16, 6))
    import datetime
    from ibats_utils.mess import datetime_2_str
    plt.suptitle(datetime_2_str(datetime.datetime.now()))
    plt.show()
    value_df = pd.DataFrame({num: df['value']
                             for num, df in enumerate(episodes_train, start=1)
                             if df.shape[0] > 0})
    value_df.plot()
    title = datetime_2_str(datetime.datetime.now())
    plt.suptitle(title)
    from ibats_common.analysis.plot import plot_or_show
    plot_or_show(enable_save_plot=True, enable_show_plot=True, file_name=f'train_{title}')

    path = agent.save_model()
    print('model save to path:', path)
    agent.close()
    return reward_df


def _test_agent2():
    import pandas as pd
    import numpy as np
    # 建立相关数据
    n_step = 250
    ohlcav_col_name_list = ["open", "high", "low", "close", "amount", "volume"]
    from ibats_common.example.data import load_data
    md_df = load_data('RB.csv').set_index('trade_date')[ohlcav_col_name_list]
    md_df.index = pd.DatetimeIndex(md_df.index)
    from ibats_common.backend.factor import get_factor, transfer_2_batch
    factors_df = get_factor(md_df, dropna=True)
    df_index, df_columns, data_arr_batch = transfer_2_batch(factors_df, n_step=n_step)
    shape = [data_arr_batch.shape[0], 5, int(n_step / 5), data_arr_batch.shape[2]]
    batch_factors = np.transpose(data_arr_batch.reshape(shape), [0, 2, 3, 1])
    md_df = md_df.loc[df_index, :]
    print(data_arr_batch.shape, '->', shape, '->', batch_factors.shape)

    success_count, success_max_count, round_n = 0, 10, 0
    while True:
        round_n += 1
        df = train(md_df, batch_factors, round_n=round_n)
        print(df.iloc[-1, :])
        if df["value"].iloc[-1] > df["value"].iloc[0]:
            success_count += 1
            print('is win %d/%d' % (success_count, success_max_count))
            if success_count >= success_max_count:
                break


def load_predict(md_df, data_factors, tail_n=1):
    import pandas as pd
    from ibats_common.backend.rl.emulator.account import Account
    latest_state = data_factors[-1:]
    env = Account(md_df, data_factors=data_factors,expand_dims=False)
    agent = Agent(input_shape=data_factors.shape)
    path = "model/ddqn.ckpt"
    agent.restore_model(path=path)
    action = agent.choose_action_deterministic(latest_state)
    print('latest action', action)


def _test_load_predict():
    import pandas as pd
    import numpy as np
    # 建立相关数据
    n_step = 250
    ohlcav_col_name_list = ["open", "high", "low", "close", "amount", "volume"]
    from ibats_common.example.data import load_data
    md_df = load_data('RB.csv').set_index('trade_date')[ohlcav_col_name_list]
    md_df.index = pd.DatetimeIndex(md_df.index)
    from ibats_common.backend.factor import get_factor, transfer_2_batch
    factors_df = get_factor(md_df, dropna=True)
    df_index, df_columns, data_arr_batch = transfer_2_batch(factors_df, n_step=n_step)
    shape = [data_arr_batch.shape[0], 5, int(n_step / 5), data_arr_batch.shape[2]]
    data_factors = np.transpose(data_arr_batch.reshape(shape), [0, 2, 3, 1])
    md_df = md_df.loc[df_index, :]
    print(data_arr_batch.shape, '->', shape, '->', data_factors.shape)

    load_predict(md_df, data_factors, tail_n=1)


if __name__ == '__main__':
    # _test_agent()
    _test_agent2()
    # _test_load_predict()