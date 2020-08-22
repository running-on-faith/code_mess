#! /usr/bin/env python3
"""
@author  : MG
@Time    : 2020/8/22 下午9:52
@File    : reproduce_20200623_517fb234_at_20200822_7b5636ff.py
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
    env_kwargs = {
        "reward_multiplier": -1
    }
    # num_collect_episodes 被默认设置为 epsilon_greedy 倒数的 2 背,以确保又足够的样板,防止由于随机随机策略而导致价值计算失衡
    train_drl(
        train_loop_count=500,
        num_collect_episodes=5,
        epsilon_greedy=0.1,
        train_sample_batch_size=1024,
        train_count_per_loop=10,
        gamma=1.0,
        network_kwargs_func=network_kwargs_func,
        base_path='reproduce_20200623_517fb234_at_20200822_7b5636ff',
        env_kwargs=env_kwargs,
    )
