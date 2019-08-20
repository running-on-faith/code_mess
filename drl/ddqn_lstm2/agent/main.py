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

2019-08-01
action_size 3 -> 4

2019-08-08
action_size 4 -> 2 多空 only
"""
import logging

import numpy as np
import tensorflow as tf

from ibats_common.example.drl.ddqn_lstm2.agent.framework import Framework

MODEL_NAME = 'd3qn_lstm_actoin2'
logger = logging.getLogger(__name__)


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

    def update_target(self):
        self.agent.update_target_net()

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
    env = Account(md_df, batch_factors)
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
    print(len(agent.agent.cache))
    reward_df = env.plot_data()
    reward_df.to_csv('reward_df.csv')


def get_agent(action_size=2, dueling=True, gamma=0.3, batch_size=512, memory_size=20000, epsilon_min=0.1,
              epsilon_decay=0.9990, **kwargs):
    """
    2019-08-08 尝试使用 action_size=2 多空only进行训练
    :param action_size:
    :param dueling:
    :param gamma:
    :param batch_size:
    :param memory_size:
    :param epsilon_min:
    :param epsilon_decay:
    :param kwargs:
    :return:
    """
    agent = Agent(action_size=action_size, dueling=dueling, gamma=gamma, batch_size=batch_size, memory_size=memory_size,
                  epsilon_min=epsilon_min, epsilon_decay=epsilon_decay, **kwargs)
    return agent


def train(md_df, batch_factors, round_n=0, num_episodes=400, n_episode_pre_record=40):
    import pandas as pd
    from ibats_common.backend.rl.emulator.account import Account

    # 2019-08-01
    # 降低手续费率 0.003-> 0.001
    env = Account(md_df, data_factors=batch_factors, state_with_flag=True, fee_rate=0.001)
    agent = get_agent(input_shape=batch_factors.shape)
    # num_episodes, n_episode_pre_record = 200, 20
    logs_list = []

    target_step_size = agent.update_eval_batch_size * 4
    train_step_size = agent.update_eval_batch_size

    episodes_reward_df_dic = {}
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
                # logger.debug("update_target round=%d, episode=%4d/%4d, %4d/%4d, 净值=%.4f, epsilon=%.5f",
                #              round_n, episode + 1, num_episodes, episode_step + 1,
                #              env.A.data_observation.shape[0], env.A.total_value / env.A.init_cash,
                #              agent.agent.epsilon)

            if (global_step >= train_step_size and episode_step % train_step_size == 0) or done:
                logs_list = agent.update_eval()
                # print('global_step=%d, episode_step=%d, agent.update_eval()' % (global_step, episode_step))

                if done:
                    reward_df = env.plot_data()

                    if episode % n_episode_pre_record == 0 or episode == num_episodes - 1:
                        # logger.debug("episodes_train.append round=%d, episode=%4d/%4d,"
                        #              " %4d/%4d, 净值=%.4f, epsilon=%.5f",
                        #              round_n, episode + 1, num_episodes, episode_step + 1,
                        #              env.A.data_observation.shape[0], env.A.total_value / env.A.init_cash,
                        #              agent.agent.epsilon)
                        episodes_reward_df_dic[episode] = reward_df[['value', 'value_fee0']]
                        log_str1 = f", is append to episodes_reward_df_dic[{episode}] len {len(episodes_reward_df_dic)}"
                    else:
                        log_str1 = ""

                    if episode > 0 and episode % 50 == 0:
                        # 每 50 轮，进行一次样本内测试
                        path = f"model/weights_{round_n}_{episode}.h5"
                        agent.save_model(path=path)
                        log_str2 = f", model save to path: {path}"
                    else:
                        log_str2 = ""

                    if episode > 0 and episode % 50 == 0:
                        logger.debug("done round=%d, episode=%4d/%4d, %4d/%4d, 净值=%.4f, epsilon=%.5f" +
                                     log_str1 + log_str2,
                                     round_n, episode + 1, num_episodes, episode_step + 1,
                                     env.A.data_observation.shape[0], env.A.total_value / env.A.init_cash,
                                     agent.agent.epsilon)

                    if reward_df.iloc[-1, :]['value'] == reward_df.iloc[-1, :]['value_fee0'] == env.A.init_cash:
                        logger.warning('算法优化陷入0操作陷阱，退出此轮优化')
                        break

                    break

    reward_df = env.plot_data()

    # 输出图表
    import matplotlib.pyplot as plt
    from ibats_common.analysis.plot import plot_twin
    import datetime
    from ibats_utils.mess import datetime_2_str

    dt_str = datetime_2_str(datetime.datetime.now(), '%Y-%m-%d %H%M%S')
    # 输出历史训练曲线
    if len(logs_list) > 0:
        # acc_loss_df = pd.DataFrame({'acc': acc_list, 'log(loss)': np.log(loss_list)}).ffill()
        acc_loss_df = pd.DataFrame(logs_list).ffill()
        # 对 loss 数值取 log
        do_log = True
        acc_names, loss_names = [], []
        for col_name in list(acc_loss_df.columns):
            if col_name == 'loss' or col_name.endswith('error'):
                if do_log:
                    new_col_name = f'log({col_name})'
                    acc_loss_df[new_col_name] = np.log(acc_loss_df[col_name] + 1)
                    acc_loss_df.drop(col_name, axis=1, inplace=True)
                    loss_names.append(new_col_name)
                else:
                    loss_names.append(col_name)
            else:
                acc_names.append(col_name)
        title = f'{MODEL_NAME}_train_r{round_n}_epi{num_episodes}_{dt_str}_acc_loss'
        fig = plt.figure(figsize=(12, 16))
        ax = fig.add_subplot(211)
        if len(acc_names) == 0 or len(loss_names) == 0:
            logger.error('acc_names=%s, loss_names=%s', acc_names, loss_names)
            plot_twin(acc_loss_df, None, name=title,
                      ax=ax, enable_show_plot=False, enable_save_plot=False, do_clr=False, y_scales_log=[True, False])
        else:
            plot_twin(acc_loss_df[acc_names], acc_loss_df[loss_names], name=title,
                      ax=ax, enable_show_plot=False, enable_save_plot=False, do_clr=False, y_scales_log=[False, False])
    else:
        fig = None
        ax = None

    # 展示训练结果——历史
    title = f'{MODEL_NAME}_r{round_n}_epi{num_episodes}_{dt_str}_list'
    value_df = pd.DataFrame({f'{num}_v': df['value']
                             for num, df in episodes_reward_df_dic.items()
                             if df.shape[0] > 0})
    value_fee0_df = pd.DataFrame({f'{num}_0': df['value_fee0']
                                  for num, df in episodes_reward_df_dic.items()
                                  if df.shape[0] > 0})
    if ax is not None:
        # 说明上面“历史训练曲线” 有输出图像， 因此使用 ax = fig.add_subplot(212)
        ax = fig.add_subplot(212)

    plot_twin([value_df, value_fee0_df], md_df['close'], name=title, ax=ax, folder_path='images')
    # if reward_df.iloc[-1, 0] > reward_df.iloc[0, 0]:
    path = f"model/weights_{round_n}_{num_episodes}.h5"
    agent.save_model(path=path)
    logger.debug('model save to path: %s', path)
    agent.close()
    return reward_df, path


def _test_agent2(round_from=1, round_max=40, increase=100, cpu_only=False):
    """测试模型训练过程"""
    if cpu_only:
        from ibats_common.backend.rl.utils import show_device, use_cup_only
        # devices = show_device()
        # logger.debug("%s devices len:%s", type(devices), len(devices))
        # for num, d in enumerate(devices):
        #     logger.debug("%d) %s", num, d)
        use_cup_only()

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
    batch_factors = data_arr_batch
    # shape = [data_arr_batch.shape[0], 5, int(n_step / 5), data_arr_batch.shape[2]]
    # batch_factors = np.transpose(data_arr_batch.reshape(shape), [0, 2, 3, 1])
    # print(data_arr_batch.shape, '->', shape, '->', batch_factors.shape)
    md_df = md_df.loc[df_index, :]

    # success_count, success_max_count, round_n = 0, 10, 0
    for round_n in range(round_from, round_max):
        # 执行训练
        num_episodes = 500 + round_n * increase

        df, path = train(md_df, batch_factors, round_n=round_n,
                         num_episodes=num_episodes,
                         n_episode_pre_record=int(num_episodes / 6)
                         )
        logger.debug('round_n=%d, final status:\n%s', round_n, df.iloc[-1, :])


if __name__ == '__main__':
    # _test_agent()
    _test_agent2(round_from=0, cpu_only=True)
