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
from tf_agents.policies.policy_saver import PolicySaver
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.utils import common
from dr2.common.metrics import FinalTrajectoryMetric, PlotTrajectoryMatrix, run_and_get_result
from dr2.common.env import get_env
from dr2.common.uitls import show_result, run_train_loop
from dr2.dqn20200209.train.agent import get_agent
from dr2.dqn20200209.train.policy import save_policy

logger = logging.getLogger()


def train_drl(train_loop_count=20, num_eval_episodes=1, num_collect_episodes=4,
              state_with_flag=True, eval_interval=5,
              train_count_per_loop=30, train_sample_batch_size=1024, epsilon_greedy=0.1, gamma=0.8,
              network_kwargs_func=None):
    """
    :param train_loop_count: 总体轮次数
    :param num_eval_episodes: 评估测试次数
    :param num_collect_episodes: 数据采集次数
    :param state_with_flag:带 flag 标识
    :param eval_interval:评估间隔
    :param train_count_per_loop: 每轮次的训练次数
    :param train_sample_batch_size: 每次训练提取样本数量
    :param epsilon_greedy: probability of choosing a random action in the default
        epsilon-greedy collect policy (used only if a wrapper is not provided to
        the collect_policy method).
    :param gamma: reward衰减率
    :param network_kwargs_func: network kwargs function
    :return:
    """
    logger.info("Train started")
    loop_n = 0
    env = get_env(state_with_flag=state_with_flag)
    agent = get_agent(
        env, epsilon_greedy=epsilon_greedy, gamma=gamma,
        network_kwargs_func=network_kwargs_func)
    eval_policy = agent.policy
    collect_policy = agent.collect_policy
    from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
    # collect
    collect_replay_buffer = TFUniformReplayBuffer(
        agent.collect_data_spec, env.batch_size, max_length=2500 * num_collect_episodes)
    collect_observers = [collect_replay_buffer.add_batch]
    collect_driver = DynamicEpisodeDriver(
        env, collect_policy, collect_observers, num_episodes=num_collect_episodes)
    # eval 由于历史行情相对确定,因此,获取最终rr只需要跑一次即可
    final_trajectory_rr, plot_rr = FinalTrajectoryMetric(), PlotTrajectoryMatrix()
    eval_observers = [final_trajectory_rr, plot_rr]
    eval_driver = DynamicEpisodeDriver(
        env, eval_policy, eval_observers, num_episodes=num_eval_episodes)

    run_train_loop(agent, collect_driver, eval_driver, eval_interval, num_collect_episodes,
                   train_count_per_loop, train_loop_count, train_sample_batch_size)


if __name__ == "__main__":
    train_drl(train_loop_count=200, num_collect_episodes=20, epsilon_greedy=0.1)
