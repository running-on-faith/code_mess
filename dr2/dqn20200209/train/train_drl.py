#! /usr/bin/env python3
"""
@author  : MG
@Time    : 2020/2/15 上午9:42
@File    : train_drl.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
import logging
import numpy as np
import pandas as pd
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.utils import common
from tf_agents.policies import greedy_policy
from dr2.dqn20200209.train.agent import get_agent
from dr2.dqn20200209.train.env import get_env
from dr2.dqn20200209.train.policy import get_policy

logger = logging.getLogger()


class FinalTrajectoryMetric:

    def __init__(self):
        self.replay_buffer = []
        self.final_rr = []

    def __call__(self, trajectory):
        self.replay_buffer.append(trajectory)
        if trajectory.is_last():
            pct_chgs = np.ones(len(self.replay_buffer))
            for idx, _ in enumerate(self.replay_buffer):
                pct_chgs[idx] += _.reward.numpy()
            rr = pct_chgs.prod() - 1
            self.final_rr.append(rr)
            self.replay_buffer = []

    def result(self):
        return np.mean(self.final_rr)


class PlotTrajectoryMatrix:

    def __init__(self):
        self.replay_buffer = []
        self.rr_dic = {}

    def __call__(self, trajectory):
        self.replay_buffer.append(trajectory)
        if trajectory.is_last():
            pct_chgs = np.ones(len(self.replay_buffer))
            for idx, _ in enumerate(self.replay_buffer):
                pct_chgs[idx] += _.reward.numpy()
            self.rr_dic[len(self.rr_dic)] = pct_chgs.cumprod() - 1
            self.replay_buffer = []
            # logger.info("trajectory.is_last, len(self.replay_buffer)=%d\nrr list=%s",
            #             len(self.replay_buffer), pct_chgs.cumprod() - 1)

    def result(self):
        import matplotlib.pyplot as plt
        from datetime import datetime
        from ibats_utils.mess import datetime_2_str

        # logger.info("len(self.rr_dic)=%d\n%s", len(self.rr_dic), self.rr_dic)
        if len(self.rr_dic) > 0:
            rr_df = pd.DataFrame(self.rr_dic)
            rr_df.plot()
            file_name = datetime_2_str(datetime.now()) + '.png'
            plt.savefig(file_name)
            plt.close()
            logger.debug("file_name: %s saved", file_name)
            self.replay_buffer = []
            self.rr_dic = {}
        else:
            rr_df = None
        return rr_df


def compute_rr(driver):
    driver.run()
    final_trajectory_rr = driver.observers[0]
    driver.observers[1].result()
    return final_trajectory_rr.result()


def train_drl(train_loop_count=20, num_eval_episodes=1, num_collect_episodes=4,
              state_with_flag=True, eval_interval=5,
              train_count_per_loop=10, train_sample_batch_size=1024):
    """
    :param train_loop_count: 总体轮次数
    :param num_eval_episodes: 评估测试次数
    :param num_collect_episodes: 数据采集次数
    :param state_with_flag:带 flag 标识
    :param eval_interval:评估间隔
    :param train_count_per_loop: 每轮次的训练次数
    :param train_sample_batch_size: 每次训练提取样本数量
    :return:
    """
    logger.info("Train started")
    loop_n = 0
    env = get_env(state_with_flag=state_with_flag)
    agent = get_agent(env, state_with_flag=state_with_flag)
    eval_policy = agent.policy
    collect_policy = agent.collect_policy

    from tf_agents.metrics import tf_metrics
    from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
    # collect
    collect_replay_buffer = TFUniformReplayBuffer(agent.collect_data_spec, env.batch_size)
    collect_observers = [collect_replay_buffer.add_batch]
    collect_driver = DynamicEpisodeDriver(
        env, collect_policy, collect_observers, num_episodes=num_collect_episodes)
    # eval 由于历史行情相对确定,因此,获取最终汇报只需要跑一次即可
    final_trajectory_rr, plot_rr = FinalTrajectoryMetric(), PlotTrajectoryMatrix()
    eval_observers = [final_trajectory_rr, plot_rr]
    eval_driver = DynamicEpisodeDriver(
        env, eval_policy, eval_observers, num_episodes=num_eval_episodes)

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)
    collect_driver.run = common.function(collect_driver.run)

    # Evaluate the agent's policy once before training.
    rr = compute_rr(eval_driver)
    train_step, step_last = 0, None
    rr_dic, loss_dic = {train_step: rr}, {}
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

        _loss = train_loss.loss.numpy() if train_loss else None
        loss_dic[train_step] = _loss
        if train_step % eval_interval == 0:
            rr = compute_rr(eval_driver)
            logger.info('%d/%d) train_step=%d loss=%.8f rr = %.2f%%',
                        loop_n, train_loop_count, train_step, _loss, rr * 100)
            rr_dic[train_step] = rr
        else:
            logger.info('%d/%d) train_step=%d loss=%.8f',
                        loop_n, train_loop_count, train_step, _loss)

    show_result(rr_dic, loss_dic)
    logger.info("Train finished")


def show_result(rr_dic, loss_dic):
    rr_df = pd.DataFrame([rr_dic]).T
    loss_df = pd.DataFrame([loss_dic]).T
    logger.info("rr_df\n%s", rr_df)
    logger.info("loss_df\n%s", loss_df)
    import matplotlib.pyplot as plt
    _, axes = plt.subplots(2, 1)
    rr_df.plot(ax=axes[0])
    loss_df.plot(logy=True, ax=axes[1])
    plt.show()


if __name__ == "__main__":
    train_drl(train_loop_count=150, num_collect_episodes=5)
