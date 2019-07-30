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
import logging
import numpy as np
import tensorflow as tf

from ibats_common.example.drl.d3qn_replay_2019_07_26.agent.framework import Framework


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
        self.update_eval_batch_size = self.agent.batch_size

    def choose_action_deterministic(self, inputs):
        return self.agent.get_deterministic_policy(inputs)

    def choose_action_deterministic_batch(self, inputs):
        return self.agent.get_deterministic_policy_batch(inputs)

    def choose_action_stochastic(self, inputs):
        return self.agent.get_stochastic_policy(inputs)

    def update_cache(self, state, action, reward, next_state, done):
        self.agent.update_cache(state, action, reward, next_state, done)

    def update_eval(self):
        return self.agent.update_value_net()

    def save_model(self, path="model/weights.h5"):
        # return self.saver.save(self.sess, path)
        self.agent.model_eval.save_weights(path)

    def restore_model(self, path="model/weights.h5"):
        # self.saver.restore(self.sess, path)
        self.agent.model_eval.load_weights(path)

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
    df_index, df_columns, batch_factors = transfer_2_batch(factors_df, n_step=n_step)
    md_df = md_df.loc[df_index, :]

    from ibats_common.backend.rl.emulator.account import Account
    env = Account(md_df, batch_factors, fee_rate=0.001)
    agent = Agent(input_shape=batch_factors.shape, action_size=3, dueling=True,
                  gamma=0.3, batch_size=512, memory_size=100000)
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
    print(len(agent.agent.cache_action))
    reward_df = env.plot_data()
    reward_df.to_csv('reward_df.csv')


def train(md_df, batch_factors, round_n=0, num_episodes=400, n_episode_pre_record=40, action_size=3):
    model_name = 'd3qn_reply'
    import pandas as pd
    from ibats_common.backend.rl.emulator.account import Account
    logger = logging.getLogger(__name__)
    env = Account(md_df, data_factors=batch_factors, state_with_flag=True)
    agent = Agent(input_shape=batch_factors.shape, action_size=action_size, dueling=True,
                  gamma=0.3, batch_size=512, memory_size=10000)
    # num_episodes, n_episode_pre_record = 200, 20
    acc_list, loss_list = [], []

    episodes_train = {}
    for episode in range(num_episodes):
        state = env.reset()
        episode_step = 0
        while True:
            episode_step += 1

            action = agent.choose_action_stochastic(state)
            next_state, reward, done = env.step(action)
            agent.update_cache(state, action, reward, next_state, done)
            state = next_state

            if done:
                acc_list, loss_list = agent.update_eval()
                logger.debug('round=%d, episode=%d, episode_step=%d, agent.update_eval()',
                             round_n,  episode + 1, episode_step)

                if episode % n_episode_pre_record == 0 or episode == num_episodes - 1:
                    logger.debug("done round=%d, episode=%4d/%4d, %4d/%4d, 净值=%.4f, epsilon=%.5f",
                                 round_n, episode + 1, num_episodes, episode_step + 1,
                                 env.A.data_observation.shape[0], env.A.total_value / env.A.init_cash,
                                 agent.agent.epsilon)
                    episodes_train[episode] = env.plot_data()

                if episode > 0 and episode % 10 == 0:
                    # 每 50 轮，进行一次样本内测试
                    path = f"model/weights_{round_n}_{episode}.h5"
                    agent.save_model(path=path)
                    logger.debug('model save to path: %s', path)

                break

    reward_df = env.plot_data()

    # 输出图表
    from ibats_common.analysis.plot import plot_twin, plot_or_show
    import datetime
    from ibats_utils.mess import datetime_2_str
    dt_str = datetime_2_str(datetime.datetime.now(), '%Y-%m-%d %H%M%S')
    # 输出历史训练曲线
    if len(acc_list) > 0:
        acc_loss_df = pd.DataFrame({'acc': acc_list, 'log(loss)': np.log(loss_list)}).ffill()
        title = f'{model_name}_train_r{round_n}_epi{num_episodes}_{dt_str}_acc_loss'
        plot_twin(acc_loss_df['acc'], acc_loss_df['log(loss)'], name=title)

    # 输出最后一次训练结果
    # 展示训练结果——最后一个
    # reward_df.iloc[:, 0].plot()  # figsize=(16, 6)
    # value_df = reward_df[['value', 'value_fee0']] / env.A.init_cash
    # title = f'ddqn_lstm2_r{round_n}_epi{num_episodes}_{dt_str}_last'
    # plot_twin(value_df, md_df['close'], name=title)
    # 展示训练结果——历史
    title = f'{model_name}_r{round_n}_epi{num_episodes}_{dt_str}_list'
    value_df = pd.DataFrame({f'{num}_v': df['value']
                             for num, df in episodes_train.items()
                             if df.shape[0] > 0})
    value_fee0_df = pd.DataFrame({f'{num}_0': df['value_fee0']
                                  for num, df in episodes_train.items()
                                  if df.shape[0] > 0})
    plot_twin([value_df, value_fee0_df], md_df['close'], name=title)
    # if reward_df.iloc[-1, 0] > reward_df.iloc[0, 0]:
    path = f"model/weights_{round_n}_{num_episodes}.h5"
    agent.save_model(path=path)
    logger.debug('model save to path: %s', path)
    agent.close()
    return reward_df, path


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
    # _test_agent()
    _test_agent2()
