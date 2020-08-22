#! /usr/bin/env python3
"""
@author  : MG
@Time    : 2020/8/22 下午4:16
@File    : train_20200822_39d511da_mock.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
from dr2.ddqn_20200821.train_drl import train_drl


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
    env_kwargs = {
        "is_mock_data": True,  # 使用模拟数据测试方法
    }
    # num_collect_episodes 被默认设置为 epsilon_greedy 倒数的 2 背,以确保又足够的样板,防止由于随机随机策略而导致价值计算失衡
    train_drl(
        train_loop_count=200,
        num_collect_episodes=5,
        epsilon_greedy=epsilon_greedy,
        train_sample_batch_size=1024,
        train_count_per_loop=10,
        gamma=gamma,
        network_kwargs_func=network_kwargs_func,
        base_path='train_20200822_39d511da_mock',
        env_kwargs=env_kwargs,
    )
