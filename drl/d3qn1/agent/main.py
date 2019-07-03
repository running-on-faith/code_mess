#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 2019/6/28 17:19
@File    : __init__.py.py
@contact : mmmaaaggg@163.com
@desc    :
需要安装相关包
pip install graphs dm-sonnet==1.19
https://github.com/deepmind/sonnet
dm-sonnet==1.19 对应 tensorflow==1.5.1
"""

import tensorflow as tf
from ibats_common.example.drl.d3qn1.agent.framework import Framework


class Agent(object):
    def __init__(self, input_shape=None, **kwargs):
        if input_shape is not None:
            input_shape = list(input_shape)
            input_shape[0] = None
            self.agent = Framework(input_shape=input_shape, **kwargs)
        else:
            self.agent = Framework(**kwargs)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        self.sess.graph.finalize()

    def choose_action_deterministic(self, inputs):
        return self.agent.get_deterministic_policy(self.sess, inputs)

    def choose_action_stochastic(self, inputs, epsilon=0.9):
        return self.agent.get_stochastic_policy(self.sess, inputs, epsilon)

    def update_cache(self, state, action, reward, next_state, done):
        self.agent.update_cache(state, action, reward, next_state, done)

    def update_eval(self):
        self.agent.update_value_net(self.sess)

    def update_target(self):
        self.agent.update_target_net(self.sess)

    def save_model(self, path="model/ddqn.ckpt"):
        self.saver.save(self.sess, path)

    def restore_model(self, path="model/ddqn.ckpt"):
        self.saver.restore(self.sess, path)

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
    shape = [data_arr_batch.shape[0], 5, int(n_step/5), data_arr_batch.shape[2]]
    data_factors = np.transpose(data_arr_batch.reshape(shape), [0, 2, 3, 1])
    md_df = md_df.loc[df_index, :]
    print(data_arr_batch.shape, '->', shape, '->', data_factors.shape)

    from ibats_common.backend.rl.emulator.account import Account
    env = Account(md_df, data_factors=data_factors)
    agent = Agent(input_shape=data_factors.shape)
    num_episodes = 1000

    target_step_size = 512
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

                if done and episode % 10 == 0:
                    print("episode=%d, env.A.total_value=%f" % (episode, env.A.total_value))
                    episodes_train.append(env.plot_data())
                    break

    tmp = env.plot_data()
    tmp.iloc[:, 0].plot(figsize=(16, 6))
    agent.save_model()


if __name__ == '__main__':
    # _test_agent()
    _test_agent2()
