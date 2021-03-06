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
import functools
import logging
import multiprocessing
import os
from datetime import date
import numpy as np
import pandas as pd
from ibats_common.example.data import load_data, OHLCAV_COL_NAME_LIST
from ibats_utils.mess import date_2_str

from drl import DATA_FOLDER_PATH, MODEL_SAVED_FOLDER, MODEL_ANALYSIS_IMAGES_FOLDER, MODEL_REWARDS_FOLDER, \
    TENSORBOARD_LOG_FOLDER


def train_for_n_episodes(
        md_df, batch_factors, get_agent_func, round_n=0, num_episodes=400, n_episode_pre_record=40,
        model_name=None, root_folder_path=os.path.curdir, output_reward_csv=False, env_kwargs={}, agent_kwargs={},
        valid_rate_threshold=0.6, available_check_after_n_episode=1000):
    """
    训练DRL
    保存训练参数到 models_folder_path/f"{max_date_str}_{round_n}_{episode}.h5"
    :param md_df:
    :param batch_factors:
    :param get_agent_func:
    :param round_n:
    :param num_episodes:
    :param n_episode_pre_record:
    :param model_name:
    :param root_folder_path:
    :param output_reward_csv:
    :param env_kwargs:
    :param agent_kwargs:
    :param valid_rate_threshold: 样本内预测有效数据比例阈值
    :param available_check_after_n_episode: N Episode 以后再开始进行模型有效性检查
    :return:
    """
    logger = logging.getLogger(__name__)
    root_folder_path = os.path.abspath(root_folder_path)
    models_folder_path = os.path.join(root_folder_path, MODEL_SAVED_FOLDER)
    images_folder_path = os.path.join(root_folder_path, MODEL_ANALYSIS_IMAGES_FOLDER)
    rewards_folder_path = os.path.join(root_folder_path, MODEL_REWARDS_FOLDER)
    from ibats_common.backend.rl.emulator.account import Account
    env = Account(md_df, data_factors=batch_factors, **env_kwargs)
    agent = get_agent_func(input_shape=batch_factors.shape, **agent_kwargs)
    max_date = max(md_df.index)
    max_date_str = date_2_str(max_date)
    logger.info('train until %s with env_kwargs=%s, agent_kwargs=%s', max_date_str, env_kwargs, agent_kwargs)
    # num_episodes, n_episode_pre_record = 200, 20
    logs_list = []
    episode, is_model_available, is_valid_rate_oks = 1, True, []
    episodes_nav_df_dic, model_path = {}, None
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        agent.reset_counter()
        episode_step, avg_holding_days = 0, 0
        try:
            while True:
                episode_step += 1

                action = agent.choose_action_stochastic(state)
                next_state, reward, done = env.step(action)
                agent.update_cache(state, action, reward, next_state, done)
                state = next_state

                if done:
                    # 更新 eval 网络，重新计算各个节点权重
                    logs_list = agent.update_eval()
                    # 加入 reward_df[['value', 'value_fee0']]
                    if episode > 1 and (episode % n_episode_pre_record == 0 or episode == num_episodes - 1):
                        episodes_nav_df_dic[episode] = env.plot_data()[['nav', 'nav_fee0']]
                        log_str1 = f", append to episodes_reward_df_dic[{episode}] len {len(episodes_nav_df_dic)}"
                    else:
                        log_str1 = ""

                    # 生成 .h5 模型文件
                    if episode % 50 == 0:
                        # 一卖一买算换手一次，因此你 "* 2"
                        avg_holding_days = env.A.max_step_count / env.buffer_action_count[-1] * 2
                        is_available = True
                        if avg_holding_days < 2:
                            is_available = False
                            log_str2 = f" < 2天"
                        elif avg_holding_days > 20:
                            is_available = False
                            log_str2 = f" > 20天"
                        else:
                            log_str2 = f""

                        loss_dic, valid_rate = agent.valid_model()
                        is_valid_rate_ok = valid_rate > valid_rate_threshold
                        is_valid_rate_oks.append(is_valid_rate_ok)
                        if is_valid_rate_ok:
                            log_str2 += f", 样本内测试预测有效数据比例 {valid_rate * 100:.2f}%, loss_dic={loss_dic}"
                        else:
                            is_available = False
                            log_str2 += f", 样本内测试预测有效数据比例 {valid_rate * 100:.2f}% " \
                                f"<= {valid_rate_threshold * 100:.2f}%, loss_dic={loss_dic}"

                        if is_available:
                            # 每 50 轮，进行一次样本内测试
                            model_path = os.path.join(models_folder_path, f"{max_date_str}_{round_n}_{episode}.h5")
                            model_path = agent.save_model(path=model_path)
                            if model_path is not None:
                                log_str2 = f", model save to path: {model_path}" + log_str2

                        # 输出 csv文件
                        if output_reward_csv:
                            reward_df = env.plot_data()
                            reward_df.to_csv(
                                os.path.join(rewards_folder_path, f'{max_date_str}_{round_n}_{episode}.csv'))

                    else:
                        log_str2 = ""

                    if log_str1 != "" or log_str2 != "":
                        logger.debug(
                            "train until %s, round=%d, episode=%4d/%4d, %4d/%4d, 净值=%.4f, epsilon=%7.4f%%, "
                            "action_count=%4d, 平均持仓%.2f天%s%s",
                            max_date_str, round_n, episode, num_episodes,
                            episode_step, env.A.max_step_count,
                            env.A.total_value / env.A.init_cash,
                            agent.agent.epsilon * 100, env.A.action_count,
                            env.A.max_step_count / env.A.action_count * 2, log_str1,
                            log_str2)

                    break
        except Exception as exp:
            logger.exception(
                "train until %s round=%d, episode=%4d/%4d, %4d/%4d, 净值=%.4f, epsilon=%9.5f%%, action_count=%4d",
                max_date_str, round_n, episode, num_episodes, episode_step, env.A.max_step_count,
                env.A.total_value / env.A.init_cash, agent.agent.epsilon * 100, env.buffer_action_count[-1])
            raise exp from exp
        # 一卖一买算换手一次，因此你 "* 2"
        recent_n_days = 500
        avg_holding_days = env.A.max_step_count / env.buffer_action_count[-1] * 2
        oks_of_recent_n = is_valid_rate_oks[-recent_n_days:]
        len_of_oks = len(oks_of_recent_n)
        ok_rate = np.sum(oks_of_recent_n)/len_of_oks if len_of_oks > 0 else 0
        if episode >= available_check_after_n_episode and (avg_holding_days > 20 or ok_rate < 0.5):
            # 平均持仓天数大于20，交易频度过低
            logger.warning(
                "train until %s, round=%d, episode=%4d/%4d, %4d/%4d, 净值=%.4f, epsilon=%7.4f%%, "
                "action_count=%4d, （平均持仓%5.2f > 20天 或 近%d个 episode 平均有效预测比率 %5.2f%% < 50%%），退出本次训练",
                max_date_str, round_n, episode, num_episodes,
                episode_step, env.A.max_step_count,
                env.A.total_value / env.A.init_cash,
                agent.agent.epsilon * 100, env.buffer_action_count[-1],
                avg_holding_days, recent_n_days, ok_rate * 100)
            is_model_available = False
            break

    # if reward_df.iloc[-1, 0] > reward_df.iloc[0, 0]:
    model_path = os.path.join(models_folder_path, f"{max_date_str}_{round_n}_{episode}.h5")
    if is_model_available and not os.path.exists(model_path):
        loss_dic, valid_rate = agent.valid_model()
        if valid_rate > valid_rate_threshold:
            agent.save_model(path=model_path)
            logger.debug('model save to path: %s', model_path)
        else:
            logger.exception(
                "train until %s round=%d, episode=%4d/%4d, 样本内测试预测有效数据比例 %.2f%% < %.2f%%, loss_dic=%s",
                max_date_str, round_n, episode, num_episodes, valid_rate * 100, valid_rate_threshold * 100, loss_dic)

    agent.close()

    # 生成训练结果数据
    reward_df = env.plot_data()

    # 输出图表
    # import matplotlib
    # matplotlib.use('Qt5Agg')
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
    nav_df = pd.DataFrame({f'{num}_v': df['nav']
                           for num, df in episodes_nav_df_dic.items()
                           if df.shape[0] > 0})
    nav_fee0_df = pd.DataFrame({f'{num}_0': df['nav_fee0']
                                for num, df in episodes_nav_df_dic.items()
                                if df.shape[0] > 0})
    if ax is not None:
        # 说明上面“历史训练曲线” 有输出图像， 因此使用 ax = fig.add_subplot(212)
        ax = fig.add_subplot(212)

    plot_twin([nav_df, nav_fee0_df], md_df['close'], name=title, ax=ax, folder_path=images_folder_path)

    return reward_df


def train_between_dates(md_loader_func, get_factor_func, train_round_kwargs_iter_func, range_to=None, n_step=60,
                        pool: multiprocessing.Pool = None, max_train_data_len=1000):
    """在日期范围内进行模型训练"""
    logger = logging.getLogger(__name__)
    # 建立相关数据
    # OHLCAV_COL_NAME_LIST = ["open", "high", "low", "close", "amount", "volume"]
    # md_df = load_data('RB.csv', folder_path=DATA_FOLDER_PATH, index_col='trade_date', range_to=range_to
    #                   )[OHLCAV_COL_NAME_LIST]
    md_df = md_loader_func(range_to)
    # 参数及环境设置
    range_to = max(md_df.index[md_df.index <= pd.to_datetime(range_to)]) if range_to is not None else max(md_df.index)
    # 如果 max_train_data_len 有效且数据长度过长，则求 range_from
    if max_train_data_len is not None and md_df.shape[0] > max_train_data_len > 0:
        range_from = min(md_df.sort_index().index[-max_train_data_len:])
    else:
        range_from = None

    today_str = date_2_str(date.today())
    range_to_str = date_2_str(range_to)
    root_folder_path = os.path.abspath(os.path.join(os.path.curdir, f'output{today_str}', range_to_str))
    os.makedirs(os.path.join(root_folder_path, MODEL_SAVED_FOLDER), exist_ok=True)
    os.makedirs(os.path.join(root_folder_path, MODEL_ANALYSIS_IMAGES_FOLDER), exist_ok=True)
    os.makedirs(os.path.join(root_folder_path, MODEL_REWARDS_FOLDER), exist_ok=True)

    from ibats_common.backend.factor import transfer_2_batch
    factors_df = get_factor_func(md_df)
    df_index, df_columns, batch_factors = transfer_2_batch(factors_df, n_step=n_step, date_from=range_from)
    # shape = [data_arr_batch.shape[0], 5, int(n_step / 5), data_arr_batch.shape[2]]
    # batch_factors = np.transpose(data_arr_batch.reshape(shape), [0, 2, 3, 1])
    # print(data_arr_batch.shape, '->', shape, '->', batch_factors.shape)
    md_df = md_df.loc[df_index, :]

    logger.info('开始训练，样本截止日期：%s, n_step=%d', range_to_str, n_step)
    round_list = list(train_round_kwargs_iter_func())
    round_max = len(round_list)
    result_dic = {}
    for round_n, env_kwargs, agent_kwargs, train_kwargs in round_list:
        # 执行训练
        try:
            agent_kwargs['tensorboard_log_dir'] = os.path.join(root_folder_path, f"{TENSORBOARD_LOG_FOLDER}_{round_n}")
            logger.debug('range_to=%s, round_n=%d/%d, root_folder_path=%s, agent_kwargs=%s',
                         range_to_str, round_n, round_max, root_folder_path, agent_kwargs)
            if pool is None:
                df = train_for_n_episodes(
                    md_df, batch_factors, root_folder_path=root_folder_path,
                    env_kwargs=env_kwargs, agent_kwargs=agent_kwargs, **train_kwargs)
                logger.debug('round_n=%d/%d, root_folder_path=%s, agent_kwargs=%s, final status:\n%s',
                             round_n, round_max, root_folder_path, agent_kwargs, df.iloc[-1, :])
                result_dic[round_n] = df
            else:
                result = pool.apply_async(
                    train_for_n_episodes, (md_df, batch_factors,), kwds=dict(
                        root_folder_path=root_folder_path,
                        env_kwargs=env_kwargs, agent_kwargs=agent_kwargs, **train_kwargs))
                result_dic[round_n] = result
        except ZeroDivisionError:
            pass

    return result_dic


def train_on_fix_interval_periods(md_loader_func, get_factor_func, train_round_kwargs_iter_func, base_data_count=1000,
                                  offset='1M', n_step=60, date_train_from=None, date_period_count=None,
                                  use_pool=True, max_process_count=multiprocessing.cpu_count()):
    """
    间隔指定周期进行训练
    :param md_loader_func: 数据加载器
    :param get_factor_func: 因子生成器
    :param train_round_kwargs_iter_func:训练参数迭代器函数
    :param base_data_count: 初始训练数据长度
    :param offset: 训练数据步进长度，采样间隔 D日 W周 M月 Y年
    :param n_step: 训练数据 step
    :param date_train_from: 当 date_train_from 不为空时 base_data_count 参数失效，从指定日期开始进行训练
    :param date_period_count: 以起始日期开始算起，训练几个日期的模型， 默认为 None 全部
    :param use_pool: 是否使用进程池
    :param max_process_count:最大进程数（默认 cup 数量）
    :return:
    """
    logger = logging.getLogger(__name__)
    # 建立相关数据
    # md_df = load_data('RB.csv', folder_path=DATA_FOLDER_PATH, index_col='trade_date')
    md_df = md_loader_func()
    if date_train_from is not None:
        date_min = pd.to_datetime(date_train_from)
        date_max = max(md_df.index)
    else:
        date_min, date_max = min(md_df.index[base_data_count:]), max(md_df.index)
    logger.info('加载数据，提取日期序列 [%s, %s]', date_2_str(date_min), date_2_str(date_max))
    if use_pool and max_process_count > 1:
        pool = multiprocessing.Pool(max_process_count)
    else:
        pool = None

    date_results_dict, tot_count = {}, 0
    # for date_to in pd.date_range(date_min, date_max, freq=pd.DateOffset(offset)):
    for date_period_num, (date_to, data_count) in enumerate(
            md_df[(date_min <= md_df.index) & (md_df.index <= date_max)]['close'].resample(offset).count().items(),
            start=1):
        if data_count <= 0 or (False if date_period_count is None else (date_period_num > date_period_count)):
            continue
        round_result_dic = train_between_dates(
            md_loader_func, get_factor_func, train_round_kwargs_iter_func=train_round_kwargs_iter_func,
            range_to=date_to, n_step=n_step, pool=pool)
        if use_pool:
            date_results_dict[date_to] = round_result_dic
            tot_count += len(round_result_dic)

    # 池化操作，异步运行，因此需要在程序结束前阻塞，等待所有进程运行结束
    if use_pool:
        finished_count, error_count = 0, 0
        for num, date_to in enumerate(list(date_results_dict.keys()), start=1):
            round_result_dic = date_results_dict[date_to]
            for round_n in list(round_result_dic.keys()):
                result = round_result_dic[round_n]
                try:
                    df = result.get()
                    finished_count += 1
                    logger.debug('%d/%d) %s -> %d  执行结束，累计执行成功 %d 个，执行失败 %d 个，当前任务执行最终状态:\n%s',
                                 finished_count + error_count, tot_count, date_2_str(date_to), round_n,
                                 finished_count, error_count, df.iloc[-1, :])
                except Exception as exp:
                    error_count += 1
                    logger.exception("%d/%d) %s -> %d 执行异常，累计执行成功 %d 个，执行失败 %d 个",
                                     finished_count + error_count, tot_count, date_2_str(date_to), round_n,
                                     finished_count, error_count)
                    if isinstance(exp, KeyboardInterrupt):
                        raise exp from exp

                # 释放内存
                if round_n is not None:
                    del round_result_dic[round_n]

    logger.info('所有任务完成')


def _test_train_on_each_period():
    from drl.d3qnr20191127.train_drl import train_round_iter_func
    from ibats_common.backend.factor import get_factor
    from ibats_common.example import get_trade_date_series, get_delivery_date_series
    instrument_type = 'RB'
    trade_date_series = get_trade_date_series(DATA_FOLDER_PATH)
    delivery_date_series = get_delivery_date_series(instrument_type, DATA_FOLDER_PATH)
    get_factor_func = functools.partial(get_factor,
                                        trade_date_series=trade_date_series, delivery_date_series=delivery_date_series)

    train_on_fix_interval_periods(
        md_loader_func=lambda range_to=None: load_data(
            f'{instrument_type}.csv', folder_path=DATA_FOLDER_PATH, index_col='trade_date', range_to=range_to
        )[OHLCAV_COL_NAME_LIST],
        get_factor_func=get_factor_func,
        train_round_kwargs_iter_func=train_round_iter_func(2),
        use_pool=True,
        date_period_count=1,  # None 如果需要训练全部日期
    )


def _test_train_on_range(use_pool=False):
    """测试 train_between_dates """
    from drl.d3qnr20191127.train_drl import train_round_iter_func
    logger = logging.getLogger(__name__)
    base_data_count, n_step, max_process_count = 1000, 60, multiprocessing.cpu_count() // 2
    date_to = pd.to_datetime('2014-11-04')
    instrument_type = 'RB'
    md_loader_func = lambda range_to=None: load_data(
        f'{instrument_type}.csv', folder_path=DATA_FOLDER_PATH, index_col='trade_date', range_to=range_to
    )[OHLCAV_COL_NAME_LIST]
    if use_pool:
        pool = multiprocessing.Pool(2)
    else:
        pool = None
    results = {}

    from ibats_common.backend.factor import get_factor
    from ibats_common.example import get_trade_date_series, get_delivery_date_series
    trade_date_series = get_trade_date_series(DATA_FOLDER_PATH)
    delivery_date_series = get_delivery_date_series(instrument_type, DATA_FOLDER_PATH)
    get_factor_func = functools.partial(get_factor,
                                        trade_date_series=trade_date_series, delivery_date_series=delivery_date_series)
    result_dic = train_between_dates(
        md_loader_func=md_loader_func, get_factor_func=get_factor_func,
        train_round_kwargs_iter_func=functools.partial(train_round_iter_func, round_n_per_target_day=2),
        range_to=date_to, n_step=n_step, pool=pool)
    if use_pool:
        results[date_to] = result_dic

    # 池化操作，异步运行，因此需要在程序结束前阻塞，等待所有进程运行结束
    if use_pool:
        while True:
            date_to, result_dic = None, {}
            for date_to, result_dic in results.items():
                round_n = None
                for round_n, result in result_dic.items():
                    try:
                        df = result.get()
                        logger.debug('%s -> %d  执行结束, final status:\n%s', date_2_str(date_to), round_n, df.iloc[-1, :])
                    except Exception as exp:
                        if isinstance(exp, KeyboardInterrupt):
                            raise exp from exp
                        logger.exception("%s -> %d 执行异常", date_2_str(date_to), round_n)

                    break

                if round_n is not None:
                    del result_dic[round_n]

                break

            if date_to is not None and len(result_dic) == 0:
                logger.info('%s 所有任务完成', date_2_str(date_to))
                del results[date_to]

            if len(results) == 0:
                logger.info('所有任务完成')
                break


if __name__ == '__main__':
    _test_train_on_range(use_pool=False)
    # _test_train_on_each_period()
