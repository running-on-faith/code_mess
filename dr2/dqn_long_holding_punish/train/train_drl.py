#! /usr/bin/env python3
"""
@author  : MG
@Time    : 2020/7/29 下午9:35
@File    : train_drl.py
@contact : mmmaaaggg@163.com
@desc    : 实验在 long holding punish 的条件下如何构建强化学习网络
"""
import json
import logging
import os
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from dr2.common.metrics import FinalTrajectoryMetric, PlotTrajectoryMatrix
from dr2.common.env import get_env
from dr2.common.uitls import run_train_loop
from dr2.dqn_long_holding_punish.train.agent import get_agent

logger = logging.getLogger()


def train_drl(train_loop_count=20, num_eval_episodes=1, num_collect_episodes=4,
              state_with_flag=True, eval_interval=5,
              train_count_per_loop=30, train_sample_batch_size=1024, epsilon_greedy=0.1, gamma=0.8,
              network_kwargs_func=None, record_params=True, base_path=None):
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
    :param record_params: 记录参数道文件:
    :param base_path: 安装key_path分目录保存训练结果及参数
    :return:
    """
    logger.info("Train started")
    env = get_env(state_with_flag=state_with_flag, long_holding_punish=5, punish_value=0.01)
    agent, agent_kwargs = get_agent(
        env, epsilon_greedy=epsilon_greedy, gamma=gamma,
        network_kwargs_func=network_kwargs_func)
    if record_params:
        # 记录本次执行程序的参数
        train_params = {
            "train_loop_count": train_loop_count,
            "num_eval_episodes": num_eval_episodes,
            "num_collect_episodes": num_collect_episodes,
            "state_with_flag": state_with_flag,
            "eval_interval": eval_interval,
            "train_count_per_loop": train_count_per_loop,
            "train_sample_batch_size": train_sample_batch_size,
            "agent_kwargs": agent_kwargs,
        }

        def json_default_func(obj):
            if isinstance(obj, set):
                return list(obj)

        if base_path is not None:
            os.makedirs(base_path, exist_ok=True)
        else:
            base_path = os.path.curdir

        params_file_path = os.path.join(base_path, "params.json")
        with open(params_file_path, 'w') as f:
            json.dump(train_params, f, default=json_default_func, indent=4)

    eval_policy = agent.policy
    collect_policy = agent.collect_policy
    # collect
    collect_replay_buffer = TFUniformReplayBuffer(
        agent.collect_data_spec, env.batch_size, max_length=2500 * num_collect_episodes)
    collect_observers = [collect_replay_buffer.add_batch]
    collect_driver = DynamicEpisodeDriver(
        env, collect_policy, collect_observers, num_episodes=num_collect_episodes)
    # eval 由于历史行情相对确定,因此,获取最终rr只需要跑一次即可
    final_trajectory_rr, plot_rr = FinalTrajectoryMetric(), PlotTrajectoryMatrix(base_path=base_path)
    eval_observers = [final_trajectory_rr, plot_rr]
    eval_driver = DynamicEpisodeDriver(
        env, eval_policy, eval_observers, num_episodes=num_eval_episodes)

    run_train_loop(agent, collect_driver, eval_driver, eval_interval, num_collect_episodes,
                   train_count_per_loop, train_loop_count, train_sample_batch_size, base_path)


if __name__ == "__main__":
    train_drl(train_loop_count=200, num_collect_episodes=20, epsilon_greedy=0.1)
