#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 19-9-3 上午11:15
@File    : trainer.py
@contact : mmmaaaggg@163.com
@desc    : 用于进行指定日期范围数据训练
在 drl/d3qn_replay_2019_08_25/agent/main.py 基础上改进
"""
import logging
import os

import numpy as np
import pandas as pd
from ibats_utils.mess import date_2_str

from drl import DATA_FOLDER_PATH


def train(md_df, batch_factors, get_agent_func, round_n=0, num_episodes=400, n_episode_pre_record=40,
          model_name=None, root_folder_path=os.path.curdir, env_kwargs={}, agent_kwargs={}):
    logger = logging.getLogger(__name__)
    root_folder_path = os.path.abspath(root_folder_path)
    models_folder_path = os.path.join(root_folder_path, 'model')
    images_folder_path = os.path.join(root_folder_path, 'images')
    rewards_folder_path = os.path.join(root_folder_path, 'rewards')
    from ibats_common.backend.rl.emulator.account import Account
    env = Account(md_df, data_factors=batch_factors, **env_kwargs)
    agent = get_agent_func(input_shape=batch_factors.shape, **agent_kwargs)
    max_date = max(md_df.index)
    max_date_str = date_2_str(max_date)
    logger.info('train until %s with env_kwargs=%s, agent_kwargs=%s', max_date_str, env_kwargs, agent_kwargs)
    # num_episodes, n_episode_pre_record = 200, 20
    logs_list = []
    is_broken = False
    episodes_reward_df_dic, model_path = {}, None
    for episode in range(num_episodes):
        state = env.reset()
        episode_step = 0
        try:
            while True:
                episode_step += 1

                action = agent.choose_action_stochastic(state)
                next_state, reward, done = env.step(action)
                agent.update_cache(state, action, reward, next_state, done)
                state = next_state

                if done:

                    logs_list = agent.update_eval()

                    if episode % n_episode_pre_record == 0 or episode == num_episodes - 1:
                        episodes_reward_df_dic[episode] = env.plot_data()[['value', 'value_fee0']]
                        log_str1 = f", append to episodes_reward_df_dic[{episode}] len {len(episodes_reward_df_dic)}"
                    else:
                        log_str1 = ""

                    if episode > 0 and episode % 50 == 0:
                        # 每 50 轮，进行一次样本内测试
                        model_path = os.path.join(models_folder_path, f"weights_{round_n}_{episode}.h5")
                        agent.save_model(path=model_path)
                        # 输出 csv文件
                        reward_df = env.plot_data()
                        reward_df.to_csv(os.path.join(rewards_folder_path, f'reward_{round_n}_{episode}.csv'))
                        # logger.debug('model save to path: %s', model_path)
                        log_str2 = f", model save to path: {model_path}"
                    else:
                        log_str2 = ""

                    if log_str1 != "" or log_str2 != "":
                        logger.debug(
                            "done round=%d, episode=%4d/%4d, %4d/%4d, 净值=%.4f, epsilon=%.5f%%, action_count=%d, "
                            "平均持仓天数 %.2f%s%s",
                            round_n,
                            episode + 1, num_episodes, episode_step + 1,
                            env.A.data_observation.shape[0], env.A.total_value / env.A.init_cash,
                            agent.agent.epsilon * 100, env.buffer_action_count[-1],
                            env.A.max_step_count / env.buffer_action_count[-1] * 2, log_str1, log_str2)

                    break
        except Exception as exp:
            logger.exception("done round=%d, episode=%4d/%4d, %4d/%4d, 净值=%.4f, epsilon=%.5f%%, action_count=%d",
                             round_n, episode + 1, num_episodes, episode_step + 1, env.A.max_step_count,
                             env.A.total_value / env.A.init_cash, agent.agent.epsilon * 100, env.buffer_action_count[-1]
                             )
            raise exp from exp
        avg_holding_days = env.A.max_step_count / env.buffer_action_count[-1] * 2  # 一卖一买算换手一次，因此你 "* 2"
        if avg_holding_days > 20:
            # 平均持仓天数大于20，交易频度过低
            logger.warning(
                "done round=%d, episode=%4d/%4d, %4d/%4d, 净值=%.4f, epsilon=%.5f%%, action_count=%d, "
                "平均持仓天数 %.2f > 20，退出本次训练",
                round_n,
                episode + 1, num_episodes, episode_step + 1, env.A.max_step_count,
                env.A.total_value / env.A.init_cash,
                agent.agent.epsilon * 100, env.buffer_action_count[-1],
                avg_holding_days)

            # if reward_df.iloc[-1, 0] > reward_df.iloc[0, 0]:
            model_path = os.path.join(models_folder_path, f"weights_{round_n}_{episode}.h5")
            agent.save_model(path=model_path)
            logger.debug('model save to path: %s', model_path)

            is_broken = True
            break

    reward_df = env.plot_data()

    # 输出图表
    import matplotlib.pyplot as plt
    from ibats_common.analysis.plot import plot_twin
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
        title = f'{model_name}_train_r{round_n}_epi{num_episodes}_{max_date_str}_acc_loss'
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
    title = f'{model_name}_r{round_n}_epi{num_episodes}_{max_date_str}_list'
    value_df = pd.DataFrame({f'{num}_v': df['value']
                             for num, df in episodes_reward_df_dic.items()
                             if df.shape[0] > 0})
    value_fee0_df = pd.DataFrame({f'{num}_0': df['value_fee0']
                                  for num, df in episodes_reward_df_dic.items()
                                  if df.shape[0] > 0})
    if ax is not None:
        # 说明上面“历史训练曲线” 有输出图像， 因此使用 ax = fig.add_subplot(212)
        ax = fig.add_subplot(212)

    plot_twin([value_df, value_fee0_df], md_df['close'], name=title, ax=ax, folder_path=images_folder_path)

    if not is_broken:
        # if reward_df.iloc[-1, 0] > reward_df.iloc[0, 0]:
        model_path = os.path.join(models_folder_path, f"weights_{round_n}_{num_episodes}.h5")
        agent.save_model(path=model_path)
        logger.debug('model save to path: %s', model_path)

    agent.close()
    return reward_df, model_path


def train_on_range(model_name, get_agent_func, range_to=None, round_from=0, round_max=5, episodes_base=2000,
                   increase=200,
                   n_step=60, env_kwargs={}, agent_kwargs={}):
    """在日期范围内进行模型训练"""
    logger = logging.getLogger(__name__)
    # 建立相关数据
    ohlcav_col_name_list = ["open", "high", "low", "close", "amount", "volume"]
    from ibats_common.example.data import load_data
    md_df = load_data('RB.csv',
                      folder_path=DATA_FOLDER_PATH, index_col='trade_date', range_to=range_to
                      )[ohlcav_col_name_list]
    # 参数及环境设置
    range_to = max(md_df.index[md_df.index <= pd.to_datetime(range_to)]) if range_to is not None else max(md_df.index)
    root_folder_path = os.path.abspath(os.path.join(os.path.curdir, 'output', date_2_str(range_to)))
    os.makedirs(os.path.join(root_folder_path, 'model'), exist_ok=True)
    os.makedirs(os.path.join(root_folder_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(root_folder_path, 'rewards'), exist_ok=True)

    from ibats_common.backend.factor import get_factor, transfer_2_batch
    factors_df = get_factor(md_df, dropna=True)
    df_index, df_columns, data_arr_batch = transfer_2_batch(factors_df, n_step=n_step)
    batch_factors = data_arr_batch
    # shape = [data_arr_batch.shape[0], 5, int(n_step / 5), data_arr_batch.shape[2]]
    # batch_factors = np.transpose(data_arr_batch.reshape(shape), [0, 2, 3, 1])
    # print(data_arr_batch.shape, '->', shape, '->', batch_factors.shape)
    md_df = md_df.loc[df_index, :]

    logger.debug('开始训练，')
    # success_count, success_max_count, round_n = 0, 10, 0
    for round_n in range(round_from, round_max + 1):
        # 执行训练
        num_episodes = episodes_base + round_n * increase
        try:
            agent_kwargs['tensorboard_log_dir'] = os.path.join(root_folder_path, 'log')
            logger.debug('round_n=%d/%d, root_folder_path=%s, agent_kwargs=%s',
                         round_n, round_max, root_folder_path, agent_kwargs)
            df, path = train(md_df, batch_factors, get_agent_func, round_n=round_n,
                             num_episodes=num_episodes, model_name=model_name,
                             n_episode_pre_record=int(num_episodes / 6),
                             root_folder_path=root_folder_path,
                             env_kwargs=env_kwargs, agent_kwargs=agent_kwargs)
            logger.debug('round_n=%d/%d, root_folder_path=%s, agent_kwargs=%s, final status:\n%s',
                         round_n, round_max, root_folder_path, agent_kwargs, df.iloc[-1, :])
        except ZeroDivisionError:
            pass


def train_on_each_period(model_name, get_agent_func, base_data_count=1000, offset=180, batch_size=512, **train_kwargs):
    """间隔指定周期进行训练"""
    logger = logging.getLogger(__name__)
    env_kwargs = dict(state_with_flag=True, fee_rate=0.001)
    agent_kwargs = dict(batch_size=batch_size, epsilon_memory_size=20)
    # 建立相关数据
    from ibats_common.example.data import load_data
    md_df = load_data('RB.csv',
                      folder_path=DATA_FOLDER_PATH, index_col='trade_date'
                      )
    logger.info('加载数据，提取日期序列')
    date_min, date_max = min(md_df.index[base_data_count:]), max(md_df.index[:-30])
    md_df = None  # 释放内存
    for date_to in pd.date_range(date_min, date_max, freq=pd.DateOffset(offset)):
        logger.info('开始训练，样本截止日期：%s', date_to)
        train_on_range(model_name=model_name, get_agent_func=get_agent_func, range_to=date_to,
                       env_kwargs=env_kwargs, agent_kwargs=agent_kwargs, **train_kwargs)


if __name__ == '__main__':
    from drl.d3qn_replay_2019_08_25.agent.main import MODEL_NAME, get_agent

    train_on_each_period(model_name=MODEL_NAME, get_agent_func=get_agent, round_from=1, round_max=4)
