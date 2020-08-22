#! /usr/bin/env python3
"""
@author  : MG
@Time    : 2020/8/22 下午3:10
@File    : train_reproduce_20200623 517fb234.py
@contact : mmmaaaggg@163.com
@desc    : 重现 20200623 517fb234 的训练结果
"""
from dr2.dqn20200209.train.train_drl import train_drl


def network_kwargs_func(input_tensor_spec, action_spec):
    """
        Create network kwargs
        Args:
          input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
            input observations.
          action_spec: A nest of `tensor_spec.BoundedTensorSpec` representing the
            actions.
          recurrent_dropout: a float number within range [0, 1). The ratio that the
            recurrent state weights need to dropout.
    """
    state_spec = input_tensor_spec[0]
    input_shape = state_spec.shape[-1]
    network_kwargs = {
        "lstm_kwargs": {
            "dropout": 0.2,
            "recurrent_dropout": 0.2,
        },
        "fc_layer_params": [100],
        'batch_normalization': False,
        "activation_fn": "tanh"
    }
    return network_kwargs


if __name__ == "__main__":
    epsilon_greedy = 0.1
    gamma = 1.0
    # num_collect_episodes 被默认设置为 epsilon_greedy 倒数的 2 背,以确保又足够的样板,防止由于随机随机策略而导致价值计算失衡
    train_drl(
        train_loop_count=200,
        num_collect_episodes=5,
        epsilon_greedy=epsilon_greedy,
        train_sample_batch_size=1024,
        train_count_per_loop=10,
        gamma=gamma,
        network_kwargs_func=network_kwargs_func,
        base_path='reproduce_20200623_517fb234_at_20200822_07096044'
    )
