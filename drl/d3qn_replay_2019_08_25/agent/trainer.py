#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 2019/9/2 20:55
@File    : trainer.py
@contact : mmmaaaggg@163.com
@desc    : 用于进行指定日期范围数据训练
`在 drl/d3qn_replay_2019_08_25/agent/main.py 基础上改进
"""
if True:
    from ibats_common.backend.rl.utils import use_cup_only
    use_cup_only()

import logging
import os
import numpy as np

from drl.d3qn_replay_2019_08_25.agent.main import get_agent

logger = logging.getLogger(__name__)


def train(md_df, batch_factors, round_n=0, num_episodes=400, n_episode_pre_record=40, target_episode_size=20,
          root_folder_path=os.path.curdir, env_kwargs={}, agent_kwargs={}):
    models_folder_path = os.path.join(root_folder_path, 'model')
    images_folder_path = os.path.join(root_folder_path, 'images')
    rewards_folder_path = os.path.join(root_folder_path, 'rewards')
    import pandas as pd
    from ibats_common.backend.rl.emulator.account import Account
    env = Account(md_df, data_factors=batch_factors, **env_kwargs)
    agent = get_agent(input_shape=batch_factors.shape, **agent_kwargs)
    logger.info('train params env_kwargs=%s, agent_kwargs=%s', env_kwargs, agent_kwargs)
    # num_episodes, n_episode_pre_record = 200, 20
    logs_list = []

    episodes_reward_df_dic = {}
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

                    if episode % target_episode_size == 0:
                        agent.update_target()
                        log_str0 = ', update_target'
                    else:
                        log_str0 = ""

                    # logger.debug('agent.update_eval()')
                    logs_list = agent.update_eval()
                    # logger.debug('round=%d, episode=%d, episode_step=%d, agent.update_eval()',
                    #              round_n,  episode + 1, episode_step)

                    if episode % n_episode_pre_record == 0 or episode == num_episodes - 1:
                        episodes_reward_df_dic[episode] = env.plot_data()[['value', 'value_fee0']]
                        log_str1 = f", append to episodes_reward_df_dic[{episode}] len {len(episodes_reward_df_dic)}"
                    else:
                        log_str1 = ""

                    if episode > 0 and episode % 50 == 0:
                        # 每 50 轮，进行一次样本内测试
                        path = os.path.join(models_folder_path, f"weights_{round_n}_{episode}.h5")
                        agent.save_model(path=path)
                        # 输出 csv文件
                        reward_df = env.plot_data()
                        reward_df.to_csv(os.path.join(rewards_folder_path, f'reward_{round_n}_{episode}.csv'))
                        # logger.debug('model save to path: %s', path)
                        log_str2 = f", model save to path: {path}"
                    else:
                        log_str2 = ""

                    if log_str0 != "" or log_str1 != "" or log_str2 != "":
                        logger.debug(
                            "done round=%d, episode=%4d/%4d, %4d/%4d, 净值=%.4f, epsilon=%.5f%%, action_count=%d%s%s%s",
                            round_n, episode + 1, num_episodes, episode_step + 1,
                            env.A.data_observation.shape[0], env.A.total_value / env.A.init_cash,
                                     agent.agent.epsilon * 100, env.buffer_action_count[-1], log_str0, log_str1,
                            log_str2)

                    break
        except Exception as exp:
            logger.exception("done round=%d, episode=%4d/%4d, %4d/%4d, 净值=%.4f, epsilon=%.5f%%, action_count=%d",
                             round_n, episode + 1, num_episodes, episode_step + 1,
                             env.A.data_observation.shape[0], env.A.total_value / env.A.init_cash,
                             agent.agent.epsilon * 100, env.buffer_action_count[-1])
            raise exp from exp
        # 加入 action 指令变化小于 10 次，则视为训练无效，退出当期训练，重新训练
        if env.buffer_action_count[-1] < 10:
            logger.warning(
                "done round=%d, episode=%4d/%4d, %4d/%4d, 净值=%.4f, epsilon=%.5f%%, action_count=%d < 10，退出本次训练",
                round_n, episode + 1, num_episodes, episode_step + 1,
                env.A.data_observation.shape[0], env.A.total_value / env.A.init_cash,
                         agent.agent.epsilon * 100, env.buffer_action_count[-1])
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

    plot_twin([value_df, value_fee0_df], md_df['close'], name=title, ax=ax, folder_path=images_folder_path)

    # if reward_df.iloc[-1, 0] > reward_df.iloc[0, 0]:
    path = os.path.join(models_folder_path, f"weights_{round_n}_{num_episodes}.h5")
    agent.save_model(path=path)
    logger.debug('model save to path: %s', path)
    agent.close()
    return reward_df, path


def data_range_test(range_from, range_to, round_from=0, round_max=5, increase=200, batch_size=512, n_step=60):
    """测试模型训练过程"""
    import os
    from drl import DATA_FOLDER_PATH
    from ibats_utils.mess import str_2_date, date_2_str
    # 参数及环境设置
    range_from = str_2_date(range_from) if range_from is not None else None
    range_to = str_2_date(range_to) if range_to is not None else None
    root_folder_path = os.path.join(os.path.curdir, date_2_str(range_to))
    os.makedirs(os.path.join(root_folder_path, 'model'), exist_ok=True)
    os.makedirs(os.path.join(root_folder_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(root_folder_path, 'rewards'), exist_ok=True)

    # 建立相关数据
    ohlcav_col_name_list = ["open", "high", "low", "close", "amount", "volume"]
    from ibats_common.example.data import load_data
    md_df = load_data('RB.csv',
                      folder_path=DATA_FOLDER_PATH, index_col='trade_date', range_from=range_from, range_to=range_to
                      )[ohlcav_col_name_list]

    from ibats_common.backend.factor import get_factor, transfer_2_batch
    factors_df = get_factor(md_df, dropna=True)
    df_index, df_columns, data_arr_batch = transfer_2_batch(factors_df, n_step=n_step)
    batch_factors = data_arr_batch
    # shape = [data_arr_batch.shape[0], 5, int(n_step / 5), data_arr_batch.shape[2]]
    # batch_factors = np.transpose(data_arr_batch.reshape(shape), [0, 2, 3, 1])
    # print(data_arr_batch.shape, '->', shape, '->', batch_factors.shape)
    md_df = md_df.loc[df_index, :]

    # success_count, success_max_count, round_n = 0, 10, 0
    env_kwargs = dict(state_with_flag=True, fee_rate=0.001)
    agent_kwargs = dict(batch_size=batch_size, epsilon_memory_size=20)
    for round_n in range(round_from, round_max):
        # 执行训练
        num_episodes = 2000 + round_n * increase
        try:
            df, path = train(md_df, batch_factors, round_n=round_n,
                             num_episodes=num_episodes,
                             n_episode_pre_record=int(num_episodes / 6),
                             root_folder_path=root_folder_path,
                             env_kwargs=env_kwargs, agent_kwargs=agent_kwargs)
            logger.debug('round_n=%d, final status:\n%s', round_n, df.iloc[-1, :])
        except:
            pass


if __name__ == '__main__':
    # _test_agent()
    data_range_test(round_from=0, increase=500, batch_size=2048, n_step=120)
