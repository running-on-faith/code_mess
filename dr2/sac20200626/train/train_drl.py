#! /usr/bin/env python3
"""
@author  : MG
@Time    : 2020/2/15 上午9:42
@File    : train_drl.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
import logging
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.policies import greedy_policy
from dr2.common.metrics import FinalTrajectoryMetric, PlotTrajectoryMatrix
from dr2.common.env import get_env
from dr2.common.uitls import run_train_loop
from dr2.sac20200626.train.agent import get_agent

logger = logging.getLogger()


def train_drl(train_loop_count=20, num_eval_episodes=1, num_collect_episodes=4,
              state_with_flag=True, eval_interval=5,
              train_count_per_loop=10, train_sample_batch_size=1024,
              agent_kwargs=None):
    """
    :param train_loop_count: 总体轮次数
    :param num_eval_episodes: 评估测试次数
    :param num_collect_episodes: 数据采集次数
    :param state_with_flag:带 flag 标识
    :param eval_interval:评估间隔
    :param train_count_per_loop: 每轮次的训练次数
    :param train_sample_batch_size: 每次训练提取样本数量
    :param agent_kwargs: agent 参数 keys:
        critic_learning_rate=3e-4,
        actor_learning_rate=3e-4,
        alpha_learning_rate=3e-4,
        target_update_tau=0.005,
        target_update_period=1,
        gamma=0.99,
        reward_scale_factor=1.0,
        gradient_clipping=None,
        action_net_kwargs=None,
        critic_net_kwargs=None,
    :return:
    """
    logger.info("Train started")
    loop_n = 0
    # 建立环境
    train_env = get_env(state_with_flag=state_with_flag, is_continuous_action=True)
    eval_env = get_env(state_with_flag=state_with_flag, is_continuous_action=True)

    agent = get_agent(train_env, state_with_flag=state_with_flag,
                      **({} if agent_kwargs is None else agent_kwargs))
    eval_policy = greedy_policy.GreedyPolicy(agent.policy)
    collect_policy = agent.collect_policy

    # collect
    collect_replay_buffer = TFUniformReplayBuffer(
        agent.collect_data_spec, train_env.batch_size, max_length=2500 * num_collect_episodes)
    collect_observers = [collect_replay_buffer.add_batch]
    collect_driver = DynamicEpisodeDriver(
        train_env, collect_policy, collect_observers, num_episodes=num_collect_episodes)
    # eval 由于历史行情相对确定,因此,获取最终汇报只需要跑一次即可
    final_trajectory_rr, plot_rr = FinalTrajectoryMetric(), PlotTrajectoryMatrix()
    eval_observers = [final_trajectory_rr, plot_rr]
    eval_driver = DynamicEpisodeDriver(
        eval_env, eval_policy, eval_observers, num_episodes=num_eval_episodes)

    run_train_loop(agent, collect_driver, eval_driver, eval_interval, num_collect_episodes,
                   train_count_per_loop, train_loop_count, train_sample_batch_size)


if __name__ == "__main__":
    train_drl(train_loop_count=300, num_collect_episodes=10)
