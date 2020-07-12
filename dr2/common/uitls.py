#! /usr/bin/env python3
"""
@author  : MG
@Time    : 2020/7/12 下午9:03
@File    : uitls.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tf_agents.policies.policy_saver import PolicySaver
from tf_agents.utils import common

from dr2.common.metrics import run_and_get_result, FinalTrajectoryMetric
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


def run_train_loop(agent, collect_driver, eval_driver, eval_interval, num_collect_episodes,
                   train_count_per_loop, train_loop_count, train_sample_batch_size):
    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)
    collect_driver.run = common.function(collect_driver.run)
    collect_replay_buffer = collect_driver.observers[0].__self__

    saver = PolicySaver(eval_driver.policy)
    # Evaluate the agent's policy once before training.
    results_dic = run_and_get_result(eval_driver)
    stat_dic = results_dic[FinalTrajectoryMetric]
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
            stat_dic = results_dic[FinalTrajectoryMetric]
            logger.info('%d/%d) train_step=%d loss=%.8f rr = %.2f%% action_count = %.1f '
                        'avg_action_period = %.2f',
                        loop_n, train_loop_count, train_step, _loss, stat_dic['rr'] * 100,
                        stat_dic['action_count'], stat_dic['avg_action_period'])
            tot_stat_dic[train_step] = stat_dic
            save_policy(saver, train_step)
        else:
            logger.info('%d/%d) train_step=%d loss=%.8f',
                        loop_n, train_loop_count, train_step, _loss)
    show_result(tot_stat_dic, loss_dic)
    logger.info("Train finished")


if __name__ == "__main__":
    pass
