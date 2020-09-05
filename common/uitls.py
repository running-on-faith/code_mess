#! /usr/bin/env python3
"""
@author  : MG
@Time    : 2020/7/12 下午9:03
@File    : uitls.py
@contact : mmmaaaggg@163.com
@desc    :
"""
import logging
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tf_agents.policies.policy_saver import PolicySaver
from tf_agents.utils import common

from common.metrics import run_and_get_result, StateEpisodeRRMetric
from dr2.dqn20200209.train.policy import save_policy
logger = logging.getLogger()


def show_result(tot_stat_dic, loss_dic, file_name=None):
    """
    show
    """
    try:
        stat_df = pd.DataFrame(tot_stat_dic).T[['rr', 'action_count']]
        loss_df = pd.DataFrame([loss_dic]).T.rename(columns={0: 'loss'})
        logger.info("rr_df\n%s", stat_df)
        logger.info("loss_df\n%s", loss_df)
        _, axes = plt.subplots(2, 1)
        stat_df.plot(secondary_y=['action_count'], ax=axes[0])
        loss_df.plot(logy=True, ax=axes[1])
        if file_name is None:
            plt.show()
        else:
            plt.savefig(file_name)
    except KeyError:
        logger.exception("error")
        logger.info(tot_stat_dic)


def _interrupter_func(tot_stat_dic, min_check_len=20):
    """
    连续 20 数据点动作数量一样，或rr一致，则放弃训练
    """
    enable_save, enable_continue = True, True
    stat_df = pd.DataFrame(tot_stat_dic).T[['rr', 'action_count']]
    data_len = stat_df.shape[0]
    if data_len == 0:
        enable_save, enable_continue = False, True
    elif data_len < min_check_len:
        pass
    else:
        recent_s = stat_df['rr'].iloc[-min_check_len:, ]
        if all(recent_s == recent_s.mean()):
            logger.warning("近 %d 个周期,收益率 %s, 平局值 %f", min_check_len, recent_s, recent_s.mean())
            enable_save, enable_continue = False, False
        else:
            recent_s = stat_df['action_count'].iloc[-min_check_len:, ]
            if all(recent_s == recent_s.mean()):
                logger.warning("近 %d 个周期,交易次数 %s, 平局值 %f", min_check_len, recent_s, recent_s.mean())
                enable_save, enable_continue = False, False

    return enable_save, enable_continue


def run_train_loop(agent, collect_driver, eval_driver, eval_interval, num_collect_episodes,
                   train_count_per_loop, train_loop_count, train_sample_batch_size, base_path=None,
                   interrupter_func=_interrupter_func, enable_save_model=True):
    """
    进行循环训练
    :param agent: 总体轮次数
    :param collect_driver: 评估测试次数
    :param eval_driver: 数据采集次数
    :param eval_interval:评估间隔
    :param num_collect_episodes: 收集数据轮数
    :param train_count_per_loop: 每轮次的训练次数
    :param train_loop_count: 总训练轮数
    :param train_sample_batch_size: 每次训练提取样本数量
    :param base_path: 安装key_path分目录保存训练结果及参数
    :param interrupter_func: 训练中断器，当训练结果不及预期时、或已经达到预期是及早终止训练
    :param enable_save_model: 保存模型,默认为 True
    :return:
    """
    base_path_str = '' if base_path is None else base_path
    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)
    collect_driver.run = common.function(collect_driver.run)
    collect_replay_buffer = collect_driver.observers[0].__self__

    saver = PolicySaver(eval_driver.policy)
    # Evaluate the agent's policy once before training.
    results_dic = run_and_get_result(eval_driver)
    stat_dic = results_dic[StateEpisodeRRMetric]
    train_step, step_last = 0, None
    tot_stat_dic, loss_dic = {train_step: stat_dic}, {}
    for loop_n in range(1, train_loop_count + 1):

        # Collect a few steps using collect_policy and save to the replay buffer.
        logger.debug("%d/%d) collecting %d episodes", loop_n, train_loop_count, num_collect_episodes)
        collect_driver.run()

        # Sample a batch of data from the buffer and update the agent's network.
        database = iter(collect_replay_buffer.as_dataset(
            sample_batch_size=train_sample_batch_size,
            num_steps=agent.train_sequence_length
        ).prefetch(train_count_per_loop))
        train_loss = None
        for fetch_num, (experience, unused_info) in enumerate(database, start=1):
            try:
                try:
                    # logger.debug(
                    #     "%d/%d) train_step=%d training %d -> %d of batch_size=%d data",
                    #     loop_n, train_loop_count, train_step, fetch_num,
                    #     len(experience.observation), experience.observation[0].shape[0])
                    train_loss = agent.train(experience)
                    train_step = agent.train_step_counter.numpy()
                except Exception as exp:
                    if isinstance(exp, KeyboardInterrupt):
                        raise exp from exp
                    logger.exception("%d/%d) train error", loop_n, train_loop_count)
                    break
            except ValueError:
                logger.exception('train error')

            if fetch_num >= train_count_per_loop:
                break

        # logger.debug("clear buffer")
        collect_replay_buffer.clear()

        # logger.info("%d/%d) train_step=%d", loop_n, train_loop_count, train_step)
        if step_last is not None and step_last == train_step:
            logger.warning('keep train error. stop loop.')
            break
        else:
            step_last = train_step

        _loss = train_loss.loss.numpy() if train_loss else np.nan
        loss_dic[train_step] = _loss
        if train_step % eval_interval == 0:
            results_dic = run_and_get_result(eval_driver)
            stat_dic = results_dic[StateEpisodeRRMetric]
            logger.info('%d/%d) train_step=%d loss=%.8f rr = %.2f%% action_count = %.1f '
                        'avg_action_period = %.2f',
                        loop_n, train_loop_count, train_step, _loss, stat_dic['rr'] * 100,
                        stat_dic['action_count'], stat_dic['avg_action_period'])
            tot_stat_dic[train_step] = stat_dic
            if interrupter_func is not None:
                enable_save, enable_continue = interrupter_func(tot_stat_dic)

                # 保存结果
                if enable_save_model and enable_save:
                    save_policy(saver, train_step, base_path=base_path)

                # 终止训练
                if not enable_continue:
                    logger.error("训练结果不满足继续训练要求.退出循环. %s", base_path_str)
                    break

        else:
            logger.info('%d/%d) train_step=%d loss=%.8f',
                        loop_n, train_loop_count, train_step, _loss)

    # 训练终止，展示训练结果
    file_name = 'stat.png'
    file_path = file_name if base_path is None else os.path.join(base_path, file_name)
    show_result(tot_stat_dic, loss_dic, file_path)
    logger.info("Train of %s finished", base_path_str)


if __name__ == "__main__":
    pass
