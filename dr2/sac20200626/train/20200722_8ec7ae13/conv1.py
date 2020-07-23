#! /usr/bin/env python3
"""
@author  : MG
@Time    : 2020/7/18 下午9:25
@File    : conv1.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
from dr2.sac20200626.train.train_drl import train_drl


def _test_train(train_key):
    """
    train_key: 每一轮训练的唯一标识
    """

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
                # (0, 3, 186) -> (0, 1, 93)
                (input_shape, 3, 1, 1),
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
                # (0, 3, 186) -> (0, 1, 93)
                (input_shape, 3, 1, 1),
            },
            "activation_fn": "sigmoid"
        }
        return net_kwargs

    agent_kwargs = {
        "actor_net_kwargs_func": _actor_net_kwargs_func,
        "critic_net_kwargs_func": _critic_net_kwargs_func
    }
    train_drl(train_loop_count=300, num_collect_episodes=10, agent_kwargs=agent_kwargs,
              base_path=f"one_conv_layer_{train_key}")


if __name__ == "__main__":
    for _ in range(1, 3):
        _test_train(_)
