#! /usr/bin/env python3
"""
@author  : MG
@Time    : 2020/8/25 上午9:20
@File    : train_drl.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
import json
import logging
import os
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from dr2.common.metrics import StateEpisodeRRMetric, PlotTimeRRMatrix
from dr2.common.env import get_env
from dr2.common.uitls import run_train_loop
from dr2.dqn20200209.train.agent import get_agent

logger = logging.getLogger()


def train_drl(train_loop_count=20, num_eval_episodes=1, num_collect_episodes=4, eval_interval=5,
              train_count_per_loop=30, train_sample_batch_size=1024, epsilon_greedy=0.1, gamma=0.8,
              network_kwargs_func=None, record_params=True, base_path=None,
              env_kwargs=None, enable_save_model=True):
    """
    :param train_loop_count: 总体轮次数
    :param num_eval_episodes: 评估测试次数
    :param num_collect_episodes: 数据采集次数
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
    :param env_kwargs: Env kwargs
    :param enable_save_model: 保存模型,默认为 True
    :return:
    """
    logger.info(f"Train {base_path if base_path is not None else ''} started")
    state_with_flag = True
    env_kwargs = {} if env_kwargs is None else env_kwargs
    env_kwargs.setdefault("state_with_flag", state_with_flag)
    env = get_env(**env_kwargs)
    agent = get_agent(
        env, epsilon_greedy=epsilon_greedy, state_with_flag=state_with_flag)
    if record_params:
        # 记录本次执行程序的参数
        train_params = {
            "train_loop_count": train_loop_count,
            "num_eval_episodes": num_eval_episodes,
            "num_collect_episodes": num_collect_episodes,
            "eval_interval": eval_interval,
            "train_count_per_loop": train_count_per_loop,
            "train_sample_batch_size": train_sample_batch_size,
            "env_kwargs": env_kwargs,
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
    final_trajectory_rr, plot_rr = StateEpisodeRRMetric(), PlotTimeRRMatrix(base_path=base_path)
    eval_observers = [final_trajectory_rr, plot_rr]
    eval_driver = DynamicEpisodeDriver(
        env, eval_policy, eval_observers, num_episodes=num_eval_episodes)

    run_train_loop(agent, collect_driver, eval_driver, eval_interval, num_collect_episodes,
                   train_count_per_loop, train_loop_count, train_sample_batch_size, base_path,
                   enable_save_model=enable_save_model)


if __name__ == "__main__":
    # 保存是报错
    # ValueError: Attempted to save a function b'__inference_lstm_layer_call_fn_20957756'
    # which references a symbolic Tensor Tensor("dropout/mul_1:0", shape=(None, 93), dtype=float32)
    # that is not a simple constant. This is not supported.
    # 暂时放弃保存

    env_kwargs = {
    }
    train_drl(
        train_loop_count=500,
        num_collect_episodes=5,
        epsilon_greedy=0.1,
        train_sample_batch_size=1024,
        train_count_per_loop=10,
        base_path='train_20200825_1aa8dfec_reproduce_20200623_517fb234_2',
        env_kwargs=env_kwargs,
        enable_save_model=False,
    )
