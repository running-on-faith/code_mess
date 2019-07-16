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


def train(md_df, batch_factors, round_n=0):
    import pandas as pd
    from ibats_common.backend.rl.emulator.account import Account
    env = Account(md_df, data_factors=batch_factors, expand_dims=False)
    from ibats_common.example.drl.d3qn2.agent.ddqn import DDQN
    from ibats_common.example.drl.d3qn2.agent import parse_args
    args = parse_args(['--nb_episodes', str(200)])
    algorithm = DDQN(action_dim=3, state_dim=batch_factors.shape, args=args)

    import os
    from ibats_common.example.drl.d3qn2.utils.networks import get_session
    from keras.backend.tensorflow_backend import set_session
    sess = get_session()
    set_session(sess)
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(os.path.join(os.path.abspath(os.path.pardir), f'log{round_n}'))
    results = algorithm.train(env, args, summary_writer)

    import datetime
    from ibats_utils.mess import datetime_2_str
    title = f'd3qn2_{datetime_2_str(datetime.datetime.now())}'

    import matplotlib.pyplot as plt
    from ibats_common.analysis.plot import plot_or_show
    plt.plot([_[1] for _ in results])
    plot_or_show(enable_save_plot=True, enable_show_plot=True, file_name=f'train_mean_{title}')

    reward_df = env.plot_data()
    reward_df.iloc[:, 0].plot(figsize=(16, 6))
    plt.suptitle(title)
    plot_or_show(enable_save_plot=True, enable_show_plot=True, file_name=f'train_{title}')

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
