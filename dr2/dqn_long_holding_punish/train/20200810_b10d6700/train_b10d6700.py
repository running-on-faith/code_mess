#! /usr/bin/env python3
"""
@author  : MG
@Time    : 2020/8/8 下午9:13
@File    : train_428650b4.py
@contact : mmmaaaggg@163.com
@desc    : 
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
        "conv_layer_params": {
            # (0, 3, 186) -> (0, 3, 93)
            (input_shape, 3, 1, 1, 'same'),
            # (0, 3, 93) -> (0, 1, 186)
            (input_shape * 2, 3, 1),
        },
        "activation_fn": "sigmoid"
    }
    return network_kwargs


def test_train_env_param():
    import logging
    from multiprocessing import Pool, cpu_count
    from itertools import product, chain

    logger = logging.getLogger()
    epsilon_greedy_list = [0.1, 0.2]
    gamma_list = [0.2, 0.5, 0.8]
    punish_value_list = [0.02, 0.05, 0.1]
    with Pool(cpu_count()) as pool:
        for epsilon_greedy, gamma, long_holding_punish, punish_value in chain(
                product(epsilon_greedy_list, gamma_list, [10], punish_value_list),
                product(epsilon_greedy_list, gamma_list, [0], [0]),
        ):
            # 参数组合如下
            # 0.1 0.2 10 0.02
            # 0.1 0.2 10 0.05
            # 0.1 0.2 10 0.1
            # 0.1 0.5 10 0.02
            # 0.1 0.5 10 0.05
            # 0.1 0.5 10 0.1
            # 0.1 0.8 10 0.02
            # 0.1 0.8 10 0.05
            # 0.1 0.8 10 0.1
            # 0.2 0.2 10 0.02
            # 0.2 0.2 10 0.05
            # 0.2 0.2 10 0.1
            # 0.2 0.5 10 0.02
            # 0.2 0.5 10 0.05
            # 0.2 0.5 10 0.1
            # 0.2 0.8 10 0.02
            # 0.2 0.8 10 0.05
            # 0.2 0.8 10 0.1
            # 0.1 0.2 0 0
            # 0.1 0.5 0 0
            # 0.1 0.8 0 0
            # 0.2 0.2 0 0
            # 0.2 0.5 0 0
            # 0.2 0.8 0 0
            env_kwargs = {
                "long_holding_punish": long_holding_punish,
                "punish_value": punish_value,
            }
            base_path = f'conv2_20200810_b10d6700' \
                        f'_epsilon_greedy{int(epsilon_greedy * 10)}' \
                        f'_gamma{int(gamma * 10)}' \
                        f'_punish_value{int(punish_value * 100)}'
            kwargs = {
                "train_loop_count": 300,
                "num_collect_episodes": int(1 / epsilon_greedy),
                "epsilon_greedy": epsilon_greedy,
                "train_sample_batch_size": 1024,
                "train_count_per_loop": 10,
                "gamma": gamma,
                "network_kwargs_func": network_kwargs_func,
                "base_path": base_path,
                "env_kwargs": env_kwargs,
            }
            logger.info("%s start", base_path)
            pool.apply_async(train_drl, kwds=kwargs)

        pool.close()
        logger.info("等待全部进程结束后退出")
        pool.join()


if __name__ == "__main__":
    test_train_env_param()
