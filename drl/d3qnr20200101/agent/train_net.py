#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 19-11-29 上午9:15
@File    : train_net.py
@contact : mmmaaaggg@163.com
@desc    :
用于探测当期网络最佳参数
验证网络是否过度复杂或简单以计算当期数据
试验结果
build_model_8_layers 8层网络情况下，
recurrent_regularizer:
 l1 l2 参数在 1e-2，1e-3 之间的参数表现普遍不如更低的参数
 l1 参数效果从高到低  None =~= 1e-7 > 1e-6 > 1e-5  对于 l1-l2 参数来说 1e-7 与 None（不使用 l1_l2) 效果相当
kernel_regularizer 参数效果从高到低 None > 1e-4 > 1e-3

随着网络层数的增加，模型可以预测除有效数据的概率越来越小（可能原有是存在过度优化，导致样本外数据无法被预测）
因此可能3、4层网络更加适合当期复杂度
"""
import itertools
import logging

from drl.d3qnr20200101.agent.framework import calc_cum_reward_with_rr, \
    build_model_4_layers, build_model_3_layers, build_model_5_layers, build_model_8_layers

logger = logging.getLogger()


def get_data(train_from, valid_from, valid_to, n_step=60, action_size=2, flag_size=3, is_classification=False):
    """获取测试数据"""
    import numpy as np
    from drl import DATA_FOLDER_PATH
    from ibats_common.example.data import load_data, OHLCAV_COL_NAME_LIST
    import functools
    from ibats_common.backend.factor import get_factor
    from ibats_common.example import get_trade_date_series, get_delivery_date_series
    from ibats_common.backend.factor import transfer_2_batch
    from keras.utils import to_categorical
    instrument_type = 'RB'
    trade_date_series = get_trade_date_series(DATA_FOLDER_PATH)
    delivery_date_series = get_delivery_date_series(instrument_type, DATA_FOLDER_PATH)
    get_factor_func = functools.partial(
        get_factor, trade_date_series=trade_date_series,
        delivery_date_series=delivery_date_series)
    md_df = load_data(
        'RB.csv', folder_path=DATA_FOLDER_PATH, index_col='trade_date')[OHLCAV_COL_NAME_LIST]

    def get_xy(md_df, date_from, date_to):
        factors_df = get_factor_func(md_df)
        df_index, df_columns, batch_factors = transfer_2_batch(
            factors_df, n_step=n_step, date_from=date_from, date_to=date_to)
        pct = md_df['close'].pct_change().fillna(0)
        # calc_cum_reward_with_rr include_curr_day=False
        # 是因为在模拟环境下，当天的reward实际上是以及发生了的 pct值，因此不能被记录为未来的 reward
        # drl训练情况下，需要使用默认值 include_curr_day=True 是因为训练是的 reward是真实的 reward
        is_fit = [_ in set(df_index) for _ in md_df.index]
        size = (batch_factors.shape[0], action_size)
        rewards = calc_cum_reward_with_rr(pct, step=10, include_curr_day=False)
        # rewards = calc_cum_reward_with_calmar(pct, win_size=10, threshold=5)
        y_data = np.zeros(size)
        y_data[:, 0] = rewards[is_fit]
        y_data[:, 1] = -rewards[is_fit]
        y_data = np.concatenate([y_data, -y_data])
        if is_classification:
            y_data = to_categorical(np.argmax(y_data, axis=1), action_size)
        _flag = to_categorical(np.concatenate([
            np.zeros((size[0], 1)),
            np.ones((size[0], 1)),
        ]), flag_size)
        batch_factors = np.concatenate([batch_factors, batch_factors])
        x_data = {'state': batch_factors, 'flag': _flag}
        return x_data, y_data

    train_x, train_y = get_xy(md_df, train_from, valid_from)
    valid_x, valid_y = get_xy(md_df, valid_from, valid_to)
    return train_x, train_y, valid_x, valid_y


def try_best_params(train_from='2017-01-01', valid_from='2019-01-01',
                    valid_to='2019-04-01', n_step=60, enable_load_model=True):
    """
    测试不同层数，不同参数的网络，优化效果，以及样本内、样本外测试有效数据比例，准确率分布清空
    """
    # train_from = '2017-01-01'
    # valid_from = '2019-01-01'
    # valid_to = '2019-04-01'
    # n_step = 60
    # enable_load_model = True

    from keras.callbacks import TensorBoard
    import math
    from keras.callbacks import Callback
    import pandas as pd
    import numpy as np
    from collections import defaultdict

    class FitLog(Callback):

        def __init__(self, tag):
            super().__init__()
            self.epoch_log_dic = {}
            self.tag = tag
            self.logger = logging.getLogger(str(self.__class__))

        def on_epoch_end(self, epoch, logs=None):
            if logs is not None and len(logs) > 0:
                # self.logger.debug('%s', logs)
                self.epoch_log_dic[epoch] = logs

    action_size = 2
    flag_size = 3
    is_classification = True
    train_x, train_y, valid_x, valid_y = get_data(train_from, valid_from, valid_to, n_step, action_size, flag_size,
                                                  is_classification=is_classification)
    layer_count = 5
    params = [None, 0.0001, 0.00001, 0.000001, 0.0000001]
    param_iter = itertools.product(params, params, [None, 1e-3, 1e-4])
    param_iter = itertools.product(
        [3, 4, 5],
        itertools.chain(*itertools.repeat([[1e-7, 1e-7, None]], 14))  # layer 5 情况下最有优参数
    )
    tag_loss_dic, loop_count_dic = defaultdict(dict), {}
    for num, (layer_count, reg_params) in enumerate(param_iter, start=1):
        # logger.debug("%d) %d 层网络 %s train_x.shape=%s", num, layer_count, reg_params, train_x['state'].shape)
        # framework = Framework(
        #     input_shape=train_x['state'].shape, reg_params=reg_params,
        #     action_size=2, batch_size=512)
        # model = framework.model_eval
        if layer_count == 3:
            build_model_func = build_model_3_layers
        elif layer_count == 4:
            build_model_func = build_model_4_layers
        elif layer_count == 5:
            build_model_func = build_model_5_layers
        elif layer_count == 8:
            build_model_func = build_model_8_layers
        else:
            raise KeyError(f'layer_count = {layer_count}')
        model = build_model_func(
            input_shape=[None if num == 0 else _ for num, _ in enumerate(train_x['state'].shape)],
            flag_size=flag_size,
            learning_rate=0.001,
            reg_params=reg_params,
            action_size=action_size,
            is_classification=is_classification
        )
        # model.summary()
        loop_count_dic[layer_count] = loop_count = loop_count_dic.setdefault(layer_count, 0) + 1
        tag = f'layer{layer_count}_' \
            f'e{"e".join([str(math.floor(math.log(v, 10))) if v is not None else "_" for v in reg_params])}_' \
            f'{loop_count}'
        log_dir = f'./log_{tag}'
        tag_loss_dic[tag]['layer_count'] = layer_count
        tag_loss_dic[tag]['loop_count'] = loop_count
        import os
        model_file_path = os.path.join(log_dir, 'weights.h5')
        if enable_load_model and os.path.exists(model_file_path):
            model.load_weights(model_file_path)
            tag_loss_dic[tag]['loss'] = np.nan
        else:
            fit_log = FitLog(tag)
            model.fit(
                train_x, train_y, validation_data=(valid_x, valid_y),
                epochs=5000, batch_size=1024, verbose=0,
                callbacks=[
                    TensorBoard(log_dir=log_dir),
                    fit_log
                ],
                use_multiprocessing=True, workers=4)
            model.save_weights(model_file_path)
            df = pd.DataFrame(fit_log.epoch_log_dic).T
            tag_loss_dic[tag]['loss'] = df['loss']

        # 样本内测试模型是否有效
        predict_result = model.predict(train_x)
        if np.all(predict_result == (1 / action_size)):
            logger.warning("2%d) %d 层网络 %s 本轮训练无效", num, layer_count, tag)
            tag_loss_dic[tag]['available_rate_train'] = 0
            tag_loss_dic[tag]['available_acc_rate_train'] = np.nan
            tag_loss_dic[tag]['available_rate_valid'] = np.nan
            tag_loss_dic[tag]['available_acc_rate_valid'] = np.nan
            continue
        available_rate_train, available_acc_rate_train = calc_acc(predict_result, train_y)
        tag_loss_dic[tag]['available_rate_train'] = available_rate_train
        tag_loss_dic[tag]['available_acc_rate_train'] = available_acc_rate_train
        # df.to_csv(os.path.join(log_dir, f'{tag}.csv'))
        # 样本外测试模型是否有效
        result_y = model.predict(valid_x)
        available_rate, available_acc_rate = calc_acc(result_y, valid_y)
        logger.info("%2d) %d 层网络 %s 样本内预测, 有效结果占比%6.2f%%, 有效数据准确率%6.2f%% ; "
                    "样本外预测, 有效结果占比%6.2f%%, 有效数据准确率%6.2f%%",
                    num, layer_count, tag, available_rate_train * 100, available_acc_rate_train * 100,
                    available_rate * 100, available_acc_rate * 100)
        tag_loss_dic[tag]['available_rate_valid'] = available_rate
        tag_loss_dic[tag]['available_acc_rate_valid'] = available_acc_rate
        # logger.debug('%d) %d 层网络 %s model.predict_on_batch(valid_x)=\n%s', num, layer_count, tag, result_y)
        # 获取中间层数据
        # from keras.models import Model
        # layer = Model(model.layers[0].input, model.layers[10].output)
        if np.isnan(available_rate) or available_rate == 0:
            continue

        valid_y_4_show = np.argmax(valid_y, axis=1) if is_classification else valid_y[:, 1]
        result_y_4_show = np.argmax(result_y, axis=1) if is_classification else result_y[:, 1]
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(6, 10))  #
        ax = fig.add_subplot(311)
        ax.hist(valid_y_4_show, density=True)
        ax = fig.add_subplot(312)
        ax.hist(result_y_4_show, density=True)
        ax = fig.add_subplot(313)
        ax.plot(valid_y_4_show)
        ax.plot(result_y_4_show)
        ax.plot([(0 if _ else -1) for _ in (result_y_4_show == valid_y_4_show)])
        plt.legend(['valid_y', 'result_y', 'match'])
        plt.suptitle(f'{num}_layer{layer_count}_{tag}')
        plt.show()

    df = pd.DataFrame(tag_loss_dic).T
    df.to_csv(f'loss_layer.csv')
    import matplotlib.pyplot as plt
    # df['loss'].plot(grid=True, legend=True)
    plt.scatter(df['layer_count'], df['available_acc_rate_train'], marker='o')
    plt.scatter(df['layer_count'], df['available_acc_rate_valid'], marker='^')
    plt.show()


def calc_acc(predict_y, target_y):
    import numpy as np
    action_size = predict_y.shape[1]
    available = ~np.all(predict_y == (1 / action_size), axis=1)
    available_rate = np.sum(available) / predict_y.shape[0]
    matches = (np.argmax(predict_y, axis=1) == np.argmax(target_y, axis=1))
    available_match = matches[available]
    available_acc_rate = np.sum(available_match) / available_match.shape[0]
    return available_rate, available_acc_rate


if __name__ == "__main__":
    try_best_params()
