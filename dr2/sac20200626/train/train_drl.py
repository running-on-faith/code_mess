#! /usr/bin/env python3
"""
@author  : MG
@Time    : 2020/2/15 上午9:42
@File    : train_drl.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
import json
import logging
import os
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
              agent_kwargs=None, record_params=True, base_path=None):
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
    :param record_params: 记录参数道文件:
    :param base_path: 安装key_path分目录保存训练结果及参数
    :return:
    """
    logger.info("Train started")
    # 建立环境
    train_env = get_env(state_with_flag=state_with_flag, is_continuous_action=True)
    eval_env = get_env(state_with_flag=state_with_flag, is_continuous_action=True)

    agent, agent_kwargs = get_agent(train_env, state_with_flag=state_with_flag,
                                    **({} if agent_kwargs is None else agent_kwargs))
    if record_params:
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

        params_file_path = "params.json" if base_path is None else os.path.join(base_path, "params.json")
        with open(params_file_path, 'w') as f:
            json.dump(train_params, f, default=json_default_func, indent=4)
    eval_policy = greedy_policy.GreedyPolicy(agent.policy)
    collect_policy = agent.collect_policy

    # collect
    collect_replay_buffer = TFUniformReplayBuffer(
        agent.collect_data_spec, train_env.batch_size, max_length=2500 * num_collect_episodes)
    collect_observers = [collect_replay_buffer.add_batch]
    collect_driver = DynamicEpisodeDriver(
        train_env, collect_policy, collect_observers, num_episodes=num_collect_episodes)
    # eval 由于历史行情相对确定,因此,获取最终汇报只需要跑一次即可
    final_trajectory_rr, plot_rr = FinalTrajectoryMetric(), PlotTrajectoryMatrix(base_path)
    eval_observers = [final_trajectory_rr, plot_rr]
    eval_driver = DynamicEpisodeDriver(
        eval_env, eval_policy, eval_observers, num_episodes=num_eval_episodes)

    run_train_loop(agent, collect_driver, eval_driver, eval_interval, num_collect_episodes,
                   train_count_per_loop, train_loop_count, train_sample_batch_size, base_path)


def _test_train():
    def _actor_net_kwargs_func(observation_spec, action_spec):
        state_spec = observation_spec[0]
        input_shape = state_spec.shape[-1]
        net_kwargs = {
            "lstm_kwargs": {
                "dropout": 0.2,
                "recurrent_dropout": 0.2,
            },
            # (filters, kernel_size, stride, dilation_rate, padding)`
            # filters: Integer, the dimensionality of the output space
            #       (i.e. the number of output filters in the convolution).
            # kernel_size: An integer or tuple/list of 2 integers, specifying the
            #       height and width of the 2D convolution window.
            #       Can be a single integer to specify the same value for
            #       all spatial dimensions.
            # strides: An integer or tuple/list of 2 integers,
            #       specifying the strides of the convolution along the height and width.
            #       Can be a single integer to specify the same value for
            #       all spatial dimensions.
            #       Specifying any stride value != 1 is incompatible with specifying
            #       any `dilation_rate` value != 1.
            # dilation_rate: an integer or tuple/list of 2 integers, specifying
            #       the dilation rate to use for dilated convolution.
            #       Can be a single integer to specify the same value for
            #       all spatial dimensions.
            #       Currently, specifying any `dilation_rate` value != 1 is
            #       incompatible with specifying any stride value != 1.
            # padding: one of `"valid"` or `"same"` (case-insensitive).
            "conv_layer_params": {
                # (0, 3, 186) -> (0, 3, 93)
                (input_shape, 3, 1, 1, 'same'),
                # (0, 3, 93) -> (0, 1, 186)
                (input_shape * 2, 3, 1),
            },
            "activation_fn": "sigmoid"
        }
        return net_kwargs

    def _critic_net_kwargs_func(observation_spec, action_spec):
        state_spec = observation_spec[0]
        input_shape = state_spec.shape[-1]
        net_kwargs = {
            "lstm_kwargs": {
                "dropout": 0.2,
                "recurrent_dropout": 0.2,
            },
            # (filters, kernel_size, stride, dilation_rate, padding)`
            # filters: Integer, the dimensionality of the output space
            #       (i.e. the number of output filters in the convolution).
            # kernel_size: An integer or tuple/list of 2 integers, specifying the
            #       height and width of the 2D convolution window.
            #       Can be a single integer to specify the same value for
            #       all spatial dimensions.
            # strides: An integer or tuple/list of 2 integers,
            #       specifying the strides of the convolution along the height and width.
            #       Can be a single integer to specify the same value for
            #       all spatial dimensions.
            #       Specifying any stride value != 1 is incompatible with specifying
            #       any `dilation_rate` value != 1.
            # dilation_rate: an integer or tuple/list of 2 integers, specifying
            #       the dilation rate to use for dilated convolution.
            #       Can be a single integer to specify the same value for
            #       all spatial dimensions.
            #       Currently, specifying any `dilation_rate` value != 1 is
            #       incompatible with specifying any stride value != 1.
            # padding: one of `"valid"` or `"same"` (case-insensitive).
            "conv_layer_params": {
                # (0, 3, 186) -> (0, 3, 93)
                (input_shape, 3, 1, 1, 'same'),
                # (0, 3, 93) -> (0, 1, 186)
                (input_shape * 2, 3, 1),
            },
            "activation_fn": "sigmoid"
        }
        return net_kwargs

    agent_kwargs = {
        "actor_net_kwargs_func": _actor_net_kwargs_func,
        "critic_net_kwargs_func": _critic_net_kwargs_func
    }
    import datetime
    from ibats_utils.mess import date_2_str
    train_drl(train_loop_count=300, num_collect_episodes=10, agent_kwargs=agent_kwargs,
              base_path=date_2_str(datetime.datetime.today()))


if __name__ == "__main__":
    _test_train()
