#! /usr/bin/env python3
"""
@author  : MG
@Time    : 2020/7/25 上午11:01
@File    : train_test.py
@contact : mmmaaaggg@163.com
@desc    : 
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
    return network_kwargs


if __name__ == "__main__":
    epsilon_greedy = 0.1
    gamma = 0.8
    # num_collect_episodes 被默认设置为 epsilon_greedy 倒数的 2 背,以确保又足够的样板,防止由于随机随机策略而导致价值计算失衡
    train_drl(
        train_loop_count=500,
        num_collect_episodes=int(1 / epsilon_greedy),
        epsilon_greedy=epsilon_greedy,
        train_sample_batch_size=1024,
        train_count_per_loop=20,
        gamma=gamma,
        network_kwargs_func=network_kwargs_func,
        base_path='lstm_two_conv_sigmoid_20200726_e1cf8337'
    )
