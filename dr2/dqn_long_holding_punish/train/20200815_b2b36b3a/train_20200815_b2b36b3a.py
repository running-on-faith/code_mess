#! /usr/bin/env python3
"""
@author  : MG
@Time    : 2020/8/15 下午9:27
@File    : train_20200815_b2b36b3a.py
@contact : mmmaaaggg@163.com
@desc    : 本轮测试相较于上一轮 20200810_766dcf4b 区别在于
降低了交易费用 "fee_rate": 6e-4,  # 手续费变更为 万6  双边收取(该费用综合考虑的,手续费,划点,冲击成本)
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
            # (0, 4, 186) -> (0, 4, 93)
            (input_shape, 3, 1, 1, 'same'),
            # (0, 4, 93) -> (0, 2, 186)
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
    epsilon_greedy_list = [0.04, 0.07, 0.1]
    gamma_list = [0.2, 0.5, 0.8]
    punish_value_list = [0.02, 0.05]
    with Pool(cpu_count() // 2) as pool:
        for num, (epsilon_greedy, gamma, long_holding_punish, punish_value) in enumerate(
                chain(
                    product(epsilon_greedy_list, gamma_list, [10], punish_value_list),
                    product(epsilon_greedy_list, gamma_list, [0], [0]),
                ),
                start=1
        ):
            # 参数组合如下
            # epsilon_greedy, gamma, long_holding_punish, punish_value
            #  1) 0.04, 0.2, 10.0, 0.0
            #  2) 0.04, 0.2, 10.0, 0.1
            #  3) 0.04, 0.5, 10.0, 0.0
            #  4) 0.04, 0.5, 10.0, 0.1
            #  5) 0.04, 0.8, 10.0, 0.0
            #  6) 0.04, 0.8, 10.0, 0.1
            #  7) 0.07, 0.2, 10.0, 0.0
            #  8) 0.07, 0.2, 10.0, 0.1
            #  9) 0.07, 0.5, 10.0, 0.0
            # 10) 0.07, 0.5, 10.0, 0.1
            # 11) 0.07, 0.8, 10.0, 0.0
            # 12) 0.07, 0.8, 10.0, 0.1
            # 13) 0.10, 0.2, 10.0, 0.0
            # 14) 0.10, 0.2, 10.0, 0.1
            # 15) 0.10, 0.5, 10.0, 0.0
            # 16) 0.10, 0.5, 10.0, 0.1
            # 17) 0.10, 0.8, 10.0, 0.0
            # 18) 0.10, 0.8, 10.0, 0.1
            # 19) 0.04, 0.2, 0.0, 0.0
            # 20) 0.04, 0.5, 0.0, 0.0
            # 21) 0.04, 0.8, 0.0, 0.0
            # 22) 0.07, 0.2, 0.0, 0.0
            # 23) 0.07, 0.5, 0.0, 0.0
            # 24) 0.07, 0.8, 0.0, 0.0
            # 25) 0.10, 0.2, 0.0, 0.0
            # 26) 0.10, 0.5, 0.0, 0.0
            # 27) 0.10, 0.8, 0.0, 0.0
            # print(f'{num:2d}) {epsilon_greedy:.2f}, {gamma:.1f}, {long_holding_punish:.1f}, {punish_value:.1f}')
            env_kwargs = {
                "long_holding_punish": long_holding_punish,
                "punish_value": punish_value,
                "fee_rate": 6e-4,  # 手续费变更为 万6  双边收取(该费用综合考虑的,手续费,划点,冲击成本)
            }
            base_path = f'conv2_20200815_b2b36b3a_{num:02d}' \
                        f'_epsilon_greedy{int(epsilon_greedy * 100)}' \
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
