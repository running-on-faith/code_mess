#! /usr/bin/env python3
"""
@author  : MG
@Time    : 2020/8/16 上午10:23
@File    : train_20200816_9abc9d36.py
@contact : mmmaaaggg@163.com
@desc    : 比较当前 3 卷基层与 train_test_20200813_a504ca14 2 卷基层训练效果
"""
from dr2.dqn_long_holding_punish.train.train_drl import train_drl


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
        "conv_layer_params": (
            # (0, 4, 186) -> (0, 4, 93)
            (input_shape, 3, 1, 1, 'same'),
            # (0, 4, 93) -> (0, 2, 186)
            (input_shape * 2, 3, 1),
            # (0, 2, 186) -> (0, 1, 93)
            (input_shape, 2, 1),
        ),
        "activation_fn": "sigmoid"
    }
    return network_kwargs


if __name__ == "__main__":
    epsilon_greedy = 0.05
    gamma = 0.5
    env_kwargs = {
        "long_holding_punish": 10,
        "punish_value": 0.02,
        "fee_rate": 6e-4,  # 手续费变更为 万6  双边收取(该费用综合考虑的,手续费,划点,冲击成本)
    }
    # num_collect_episodes 被默认设置为 epsilon_greedy 倒数的 2 背,以确保又足够的样板,防止由于随机随机策略而导致价值计算失衡
    train_drl(
        train_loop_count=500,
        num_collect_episodes=int(1 / epsilon_greedy),
        epsilon_greedy=epsilon_greedy,
        train_sample_batch_size=1024,
        train_count_per_loop=10,
        gamma=gamma,
        network_kwargs_func=network_kwargs_func,
        base_path='lstm_conv3_20200816_9abc9d36',
        env_kwargs=env_kwargs,
    )
