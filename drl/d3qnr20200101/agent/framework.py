#! /usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author  : MG
@Time    : 2019/8/17 10:19
@File    : framework.py
@contact : mmmaaaggg@163.com
@desc    :
在 ibats_common/example/drl/d3qn_replay_2019_07_26/agent/framework.py 基础上增加对 reward 进行调整
将未来5步的reward，递减反向加到当期reward上面
# 选取正向奖励与负向奖励 50：50的方式(以后再实现）
在 ibats_common/example/drl/d3qn_replay_2019_08_07/agent/framework.py 基础上增加对 reward 进行调整
1）将最近的 epsilon_memory_size 轮训练集作为训练集（而不是仅用当前训练结果作为训练集）
2）训练集超过 epsilon_memory_size 时，将 tot_reward 最低者淘汰，不断保留高 tot_reward 训练集
3) 修改优化器为 Nadam： Nesterov Adam optimizer: Adam本质上像是带有动量项的RMSprop，Nadam就是带有Nesterov 动量的Adam RMSprop
2019-08-21
4) 有 random_drop_best_cache_rate 的几率 随机丢弃 cache 防止最好的用例不断累积造成过度优化
5) EpsilonMaker 提供衰减的sin波动学习曲线
7) 当 epsilon 的概率选择随机动作时，有 keep_last_action 的概率，选择延续上一个动作
2019-08-25
将eval target net分离，提高回测阶段的输出稳定性
2019-10-11
为了提高历史数据（近期数据权重）（降低远期数据权重），将历史数据进行重复叠加训练
方法：input output 安装 0.5**N 指数方式split，然后叠加，进行统一训练
2019-11-27
为了解决过度优化问他，降低网络层数以及节点数量
增加对 LSTM 层的正则化
2019-12-02
网络优化总是失败，因此又增加了3层
同时对正则化进行了参数化
2020-01-01
d3qnr20191127 方法中关于 rewards 永远存在某一列是predict预测值，另一列是实际action对应的reward。
与事情情况存在明显偏差，可能导致网络优化不足或出现片面优化的问题。
改进方法：
每一次完成 episode，将会保持对应的所有 action对应的rewards，N次episode 执行完毕后，将所有状态的各个action进行分别叠加
同状态，同action的rewards取平均值，同一状态的不同action讲有可能均有实际值，因此更能够反应真是的rewards情况
"""
import logging
import os
from typing import List

import ffn
import numpy as np
import pandas as pd

DATE_BASELINE = pd.to_datetime('2018-01-01')
DEFAULT_REG_PARAMS = [1e-7, 1e-7, None]


class EpsilonMaker:
    def __init__(self, keep_init_4_first_n=10, epsilon_decay=0.996, sin_step=0.1, epsilon_min=0.02,
                 epsilon_sin_max=0.1):
        """
        Epsilon 发生器
        :param keep_init_4_first_n: 前 N 次保持 init 值
        :param epsilon_decay: 衰减率
        :param sin_step: sin 曲线步长
        :param epsilon_min: 最小值
        :param epsilon_sin_max: sin曲线浮动最大值
        """
        self.keep_init_4_first_n = keep_init_4_first_n
        self.sin_step = sin_step
        self.sin_step_tot = 0
        self.epsilon = self.epsilon_init = 1.0  # exploration rate
        self.epsilon_sin_max = epsilon_sin_max
        self.epsilon_min = epsilon_min
        self.epsilon_down = self.epsilon  # self.epsilon_down *= self.epsilon_decay
        self.epsilon_sin = 0
        # sin 曲线上下波动高达是 2 (从 -1 到 1) 因此除以 2
        self.sin_height = (self.epsilon_sin_max - self.epsilon_min) / 2
        self.epsilon_decay = epsilon_decay
        self._count = 0

    @property
    def epsilon_next(self):
        self._count += 1
        if self._count > self.keep_init_4_first_n:
            if self.epsilon_down > self.epsilon_min:
                self.epsilon_down *= self.epsilon_decay
            self.epsilon_sin = (np.sin(self.sin_step_tot * self.sin_step) + 1) * self.sin_height
            self.sin_step_tot += 1
            self.epsilon = self.epsilon_down + self.epsilon_sin
            if self.epsilon > self.epsilon_init:
                self.epsilon = self.epsilon_init
            elif self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

        return self.epsilon


def build_model_8_layers(input_shape, flag_size, action_size, reg_params=DEFAULT_REG_PARAMS, learning_rate=0.001,
                         dueling=True, is_classification=False):
    import tensorflow as tf
    from keras.layers import Dense, LSTM, Dropout, Input, concatenate, Lambda, Activation
    from keras.models import Model
    from keras import metrics, backend
    from keras.optimizers import Nadam
    from keras.regularizers import l2, l1_l2
    # Neural Net for Deep-Q learning Model
    input_net = Input(batch_shape=input_shape, name=f'state')
    # 2019-11-27 增加对 LSTM 层的正则化
    # 根据 《使用权重症则化较少模型过拟合》，经验上，LSTM 正则化 10^-6
    # 使用 L1L2 混合正则项（又称：Elastic Net）
    recurrent_reg = l1_l2(l1=reg_params[0], l2=reg_params[1]) \
        if reg_params[0] is not None and reg_params[1] is not None else None
    kernel_reg = l2(reg_params[2]) if reg_params[2] is not None else None
    net = LSTM(
        input_shape[-1] * 2,
        recurrent_regularizer=recurrent_reg,
        kernel_regularizer=kernel_reg,
        dropout=0.3
    )(input_net)
    net = Dense(int(input_shape[-1]))(net)
    net = Dropout(0.3)(net)
    net = Dense(int(input_shape[-1] / 2))(net)
    net = Dropout(0.3)(net)
    net = Dense(int(input_shape[-1] / 4))(net)
    net = Dropout(0.3)(net)
    input2 = Input(batch_shape=[None, flag_size], name=f'flag')
    net = concatenate([net, input2])
    net = Dense((int(input_shape[-1] / 4) + flag_size) // 2)(net)
    net = Dropout(0.3)(net)
    net = Dense((int(input_shape[-1] / 4) + flag_size) // 4)(net)
    net = Dropout(0.3)(net)
    # net = Dense(self.action_size * 4, activation='relu')(net)
    if dueling:
        net = Dense(action_size + 1, activation='relu')(net)
        net = Lambda(lambda i: backend.expand_dims(i[:, 0], -1) + i[:, 1:] - backend.mean(i[:, 1:], keepdims=True),
                     output_shape=(action_size,))(net)
    else:
        net = Dense(action_size, activation='linear')(net)

    if is_classification:
        net = Activation('softmax')(net)

    model = Model(inputs=[input_net, input2], outputs=net)

    def _huber_loss(y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = backend.abs(error) <= clip_delta

        squared_loss = 0.5 * backend.square(error)
        quadratic_loss = 0.5 * backend.square(clip_delta) + clip_delta * (backend.abs(error) - clip_delta)

        return backend.mean(tf.where(cond, squared_loss, quadratic_loss))

    if is_classification:
        if action_size == 2:
            model.compile(Nadam(learning_rate), loss=_huber_loss,
                          metrics=[metrics.binary_accuracy]
                          )
        else:
            model.compile(Nadam(learning_rate), loss=_huber_loss,
                          metrics=[metrics.categorical_accuracy]
                          )
    else:
        model.compile(Nadam(learning_rate), loss=_huber_loss,
                      # metrics=[metrics.mae, metrics.mean_squared_logarithmic_error]
                      )
    # model.summary()
    return model


def build_model_5_layers(input_shape, flag_size, action_size, reg_params=DEFAULT_REG_PARAMS, learning_rate=0.001,
                         dueling=True, is_classification=False):
    import tensorflow as tf
    from keras.layers import Dense, LSTM, Dropout, Input, concatenate, Lambda, Activation
    from keras.models import Model
    from keras import metrics, backend
    from keras.optimizers import Nadam
    from keras.regularizers import l2, l1_l2
    # Neural Net for Deep-Q learning Model
    input_net = Input(batch_shape=input_shape, name=f'state')
    # 2019-11-27 增加对 LSTM 层的正则化
    # 根据 《使用权重症则化较少模型过拟合》，经验上，LSTM 正则化 10^-6
    # 使用 L1L2 混合正则项（又称：Elastic Net）

    recurrent_reg = l1_l2(l1=reg_params[0], l2=reg_params[1]) \
        if reg_params[0] is not None and reg_params[1] is not None else None
    kernel_reg = l2(reg_params[2]) if reg_params[2] is not None else None
    input_size = input_shape[-1]
    net = LSTM(
        input_size * 2,
        recurrent_regularizer=recurrent_reg,
        kernel_regularizer=kernel_reg,
        dropout=0.3
    )(input_net)
    net = Dense(int(input_size / 2))(net)
    net = Dropout(0.4)(net)
    input2 = Input(batch_shape=[None, flag_size], name=f'flag')
    net = concatenate([net, input2])
    net = Dense(int((input_size / 2 + flag_size) / 4))(net)
    net = Dropout(0.4)(net)
    if dueling:
        net = Dense(action_size + 1, activation='relu')(net)
        net = Lambda(lambda i: backend.expand_dims(i[:, 0], -1) + i[:, 1:] - backend.mean(i[:, 1:], keepdims=True),
                     output_shape=(action_size,))(net)
    else:
        net = Dense(action_size, activation='linear')(net)

    if is_classification:
        net = Activation('softmax')(net)

    model = Model(inputs=[input_net, input2], outputs=net)

    def _huber_loss(y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = backend.abs(error) <= clip_delta

        squared_loss = 0.5 * backend.square(error)
        quadratic_loss = 0.5 * backend.square(clip_delta) + clip_delta * (backend.abs(error) - clip_delta)

        return backend.mean(tf.where(cond, squared_loss, quadratic_loss))

    if is_classification:
        if action_size == 2:
            model.compile(Nadam(learning_rate), loss=_huber_loss,
                          metrics=[metrics.binary_accuracy]
                          )
        else:
            model.compile(Nadam(learning_rate), loss=_huber_loss,
                          metrics=[metrics.categorical_accuracy]
                          )
    else:
        model.compile(Nadam(learning_rate), loss=_huber_loss,
                      # metrics=[metrics.mae, metrics.mean_squared_logarithmic_error]
                      )
    # model.summary()
    return model


def build_model_4_layers(input_shape, flag_size, action_size, reg_params=DEFAULT_REG_PARAMS, learning_rate=0.001,
                         dueling=True, is_classification=False):
    import tensorflow as tf
    from keras.layers import Dense, LSTM, Dropout, Input, concatenate, Lambda, Activation
    from keras.models import Model
    from keras import metrics, backend
    from keras.optimizers import Nadam
    from keras.regularizers import l2, l1_l2
    # Neural Net for Deep-Q learning Model
    input_net = Input(batch_shape=input_shape, name=f'state')
    # 2019-11-27 增加对 LSTM 层的正则化
    # 根据 《使用权重症则化较少模型过拟合》，经验上，LSTM 正则化 10^-6
    # 使用 L1L2 混合正则项（又称：Elastic Net）
    recurrent_reg = l1_l2(l1=reg_params[0], l2=reg_params[1]) \
        if reg_params[0] is not None and reg_params[1] is not None else None
    kernel_reg = l2(reg_params[2]) if reg_params[2] is not None else None
    input_size = input_shape[-1]
    net = LSTM(
        input_size * 2,
        recurrent_regularizer=recurrent_reg,
        kernel_regularizer=kernel_reg,
        dropout=0.25
    )(input_net)
    input2 = Input(batch_shape=[None, flag_size], name=f'flag')
    net = concatenate([net, input2])
    net = Dense(int((input_size + flag_size) / 4))(net)
    net = Dropout(0.25)(net)
    if dueling:
        net = Dense(action_size + 1, activation='relu')(net)
        net = Lambda(lambda i: backend.expand_dims(i[:, 0], -1) + i[:, 1:] - backend.mean(i[:, 1:], keepdims=True),
                     output_shape=(action_size,))(net)
    else:
        net = Dense(action_size, activation='linear')(net)

    if is_classification:
        net = Activation('softmax')(net)

    model = Model(inputs=[input_net, input2], outputs=net)

    def _huber_loss(y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = backend.abs(error) <= clip_delta
        squared_loss = 0.5 * backend.square(error)
        quadratic_loss = 0.5 * backend.square(clip_delta) + clip_delta * (backend.abs(error) - clip_delta)
        return backend.mean(tf.where(cond, squared_loss, quadratic_loss))

    if is_classification:
        if action_size == 2:
            model.compile(Nadam(learning_rate), loss=_huber_loss,
                          metrics=[metrics.binary_accuracy]
                          )
        else:
            model.compile(Nadam(learning_rate), loss=_huber_loss,
                          metrics=[metrics.categorical_accuracy]
                          )
    else:
        model.compile(Nadam(learning_rate), loss=_huber_loss,
                      # metrics=[metrics.mae, metrics.mean_squared_logarithmic_error]
                      )
    # model.summary()
    return model


def build_model_3_layers(input_shape, flag_size, action_size, reg_params=DEFAULT_REG_PARAMS,
                         learning_rate=0.001, dueling=True, is_classification=False):
    import tensorflow as tf
    from keras.layers import Dense, LSTM, Input, concatenate, Lambda, Activation
    from keras.models import Model
    from keras import metrics, backend
    from keras.optimizers import Nadam
    from keras.regularizers import l2, l1_l2
    # Neural Net for Deep-Q learning Model
    input_net = Input(batch_shape=input_shape, name=f'state')
    # 2019-11-27 增加对 LSTM 层的正则化
    # 根据 《使用权重症则化较少模型过拟合》，经验上，LSTM 正则化 10^-6
    # 使用 L1L2 混合正则项（又称：Elastic Net）
    recurrent_reg = l1_l2(l1=reg_params[0], l2=reg_params[1]) \
        if reg_params[0] is not None and reg_params[1] is not None else None
    kernel_reg = l2(reg_params[2]) if reg_params[2] is not None else None
    input_size = input_shape[-1]
    net = LSTM(
        input_size * 2,
        recurrent_regularizer=recurrent_reg,
        kernel_regularizer=kernel_reg,
        dropout=0.3
    )(input_net)
    input2 = Input(batch_shape=[None, flag_size], name=f'flag')
    net = concatenate([net, input2])
    if dueling:
        net = Dense(action_size + 1, activation='relu')(net)
        net = Lambda(lambda i: backend.expand_dims(i[:, 0], -1) + i[:, 1:] - backend.mean(i[:, 1:], keepdims=True),
                     output_shape=(action_size,))(net)
    else:
        net = Dense(action_size, activation='linear')(net)

    if is_classification:
        net = Activation('softmax')(net)

    model = Model(inputs=[input_net, input2], outputs=net)

    def _huber_loss(y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = backend.abs(error) <= clip_delta

        squared_loss = 0.5 * backend.square(error)
        quadratic_loss = 0.5 * backend.square(clip_delta) + clip_delta * (backend.abs(error) - clip_delta)

        return backend.mean(tf.where(cond, squared_loss, quadratic_loss))

    if is_classification:
        if action_size == 2:
            model.compile(Nadam(learning_rate), loss=_huber_loss,
                          metrics=[metrics.binary_accuracy]
                          )
        else:
            model.compile(Nadam(learning_rate), loss=_huber_loss,
                          metrics=[metrics.categorical_accuracy]
                          )
    else:
        model.compile(Nadam(learning_rate), loss=_huber_loss,
                      # metrics=[metrics.mae, metrics.mean_squared_logarithmic_error]
                      )
    # model.summary()
    return model


class Framework(object):
    def __init__(self, input_shape=[None, 60, 93], dueling=True, action_size=4, batch_size=512,
                 learning_rate=0.001, tensorboard_log_dir='./tensorboard_log',
                 epochs=1, keep_epsilon_init_4_first_n=5, epsilon_decay=0.9990, sin_step=0.1,
                 epsilon_min=0.05, epsilon_sin_max=0.1, update_target_net_period=20,
                 cum_reward_back_step=10, epsilon_memory_size=20, target_avg_holding_days=5,
                 min_data_len_4_multiple_date=30, random_drop_cache_rate=0.01, build_model_layer_count=4,
                 reg_params=[1e-7, 1e-7, 1e-3], train_net_period=10):
        import tensorflow as tf
        from keras import backend
        from keras.callbacks import Callback

        class LogFit(Callback):

            def __init__(self):
                super().__init__()
                self.logs_list = []
                self.epsilon = 1.
                self.model_predict_unavailable_rate = 1.
                self.predict_direction_match_rate = 0.
                self.logger = logging.getLogger(str(self.__class__))

            def on_epoch_end(self, epoch, logs=None):
                if logs is not None and len(logs) > 0:
                    # self.logger.debug('%s', logs)
                    # self.loss_list.append(logs['loss'] if 'loss' in logs else np.nan)
                    # self.acc_list.append(logs['acc'] if 'acc' in logs else np.nan)
                    # self.acc_list.append(logs['categorical_accuracy'] if 'categorical_accuracy' in logs else np.nan)
                    # self.acc_list.append(logs['mean_absolute_error'] if 'mean_absolute_error' in logs else np.nan)
                    logs['epsilon'] = self.epsilon
                    logs['model_predict_unavailable_rate'] = self.model_predict_unavailable_rate
                    logs['predict_direction_match_rate'] = self.predict_direction_match_rate
                    self.logs_list.append(logs)

        self.input_shape = [None if num == 0 else _ for num, _ in enumerate(input_shape)]
        self.action_size = action_size
        if action_size <= 1:
            self.logger.error("action_size=%d, 必须大于1", action_size)
            raise ValueError(f"action_size={action_size}, 必须大于1")
        if action_size == 2:
            self.actions = [1, 2]
        else:
            self.actions = list(range(action_size))
        # actions_change_list 为action集合，形成的数组。
        # 对应每一个动作需要改变时，可以选择的动作集合。
        # 数组脚标为当前动作，对应的集合为可以选择的动作
        self.actions_change_list = [[_ for _ in self.actions if _ != action] for action in range(max(self.actions) + 1)]
        # 计数器，仅用于模型训练是记录使用
        self.last_action = None
        self.last_action_same_count = 0
        self.action_count = 0
        self.model_predict_count = 0
        self.model_predict_unavailable_count = 0
        # 延续上一执行动作的概率
        # 根据等比数列求和公式 Sn = a * (1 - q^n) / (1 - q)
        # 当 q = 1 - a 时，Sn = 1 - q^n
        # 目标平均持仓天数 N 日时 50% 概率切换动作。因此 Sn = 0.5
        # 因此 q = 0.5 ^ (1/n)
        # a = 1-0.5^(1/n)
        self.target_avg_holding_days = target_avg_holding_days
        self.target_avg_holding_rate = 1 - 0.5 ** (1 / target_avg_holding_days)
        self.cum_reward_back_step = cum_reward_back_step
        self.epsilon_memory_size = epsilon_memory_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.tensorboard_log_dir = tensorboard_log_dir
        self.dueling = dueling
        self.fit_callback = LogFit()
        # cache for experience replay
        # cache for state, action, reward, next_state, done
        self.cache_state, self.cache_action, self.cache_reward, self.cache_next_state, self.cache_done = \
            [], [], [], [], []
        self.cache_state_list, self.cache_action_list, self.cache_reward_list, self.cache_next_state_list, \
            self.cache_done_list = [], [], [], [], []
        self.logger = logging.getLogger(str(self.__class__))
        backend.clear_session()
        tf.reset_default_graph()

        self.epsilon = 1.0  # exploration rate
        # self.epsilon_min = epsilon_min
        # self.epsilon_decay = epsilon_decay
        self.epsilon_maker = EpsilonMaker(keep_epsilon_init_4_first_n, epsilon_decay, sin_step, epsilon_min,
                                          epsilon_sin_max=epsilon_sin_max)
        self.random_drop_cache_rate = random_drop_cache_rate
        self.flag_size = 3
        self.reg_params = reg_params
        self.build_model_layer_count = build_model_layer_count
        self.model_eval = self._build_model()
        self.model_target = self._build_model()
        # 标示网络是否已经被更新
        self.has_update_target_net = False
        self.epochs = epochs
        self.tot_update_count = 0
        # 每间隔多少 episode 执行一次 model.fit（训练网络）
        self.train_net_period = train_net_period
        # 每间隔多少 episode 执行一次 update_target_net （同步网络）
        self.update_target_net_period = update_target_net_period
        self.min_data_len_4_multiple_date = min_data_len_4_multiple_date

    def reset_counter(self):
        # 计数器，仅用于模型训练是记录使用
        self.last_action = None
        self.last_action_same_count = 0
        self.action_count = 0
        self.model_predict_count = 0
        self.model_predict_unavailable_count = 0

    @property
    def acc_loss_lists(self):
        """return acc_list, loss_list"""
        return self.fit_callback.logs_list

    def _build_model(self):
        if self.build_model_layer_count == 3:
            net = build_model_3_layers(
                input_shape=self.input_shape, flag_size=self.flag_size, action_size=self.action_size,
                reg_params=self.reg_params, learning_rate=self.learning_rate, dueling=self.dueling)
        elif self.build_model_layer_count == 4:
            net = build_model_4_layers(
                input_shape=self.input_shape, flag_size=self.flag_size, action_size=self.action_size,
                reg_params=self.reg_params, learning_rate=self.learning_rate, dueling=self.dueling)
        elif self.build_model_layer_count == 5:
            net = build_model_5_layers(
                input_shape=self.input_shape, flag_size=self.flag_size, action_size=self.action_size,
                reg_params=self.reg_params, learning_rate=self.learning_rate, dueling=self.dueling)
        elif self.build_model_layer_count == 8:
            net = build_model_8_layers(
                input_shape=self.input_shape, flag_size=self.flag_size, action_size=self.action_size,
                reg_params=self.reg_params, learning_rate=self.learning_rate, dueling=self.dueling)
        else:
            raise ValueError(f"build_model_layer_count={self.build_model_layer_count}")
        return net

    def get_deterministic_policy(self, inputs):
        """用于是基于模型预测使用"""
        from keras.utils import to_categorical
        # 由于 self.actions[int(np.argmax(act_values[0]))] 以及对上一个动作的 action进行过转化因此不需要再 + 1 了
        # action = inputs[1] + 1 if self.action_size == 2 else inputs[1]
        action = inputs[1]
        # self.logger.debug('flag.shape=%s, flag=%s', np.array(inputs[0]).shape, to_categorical(action, self.flag_size))
        act_values = self.model_target.predict(x={'state': np.array(inputs[0]),
                                                  'flag': to_categorical(action, self.flag_size)})
        if np.any(np.isnan(act_values)):
            self.logger.error("predict error act_values=%s", act_values)
            raise ZeroDivisionError("predict error act_values=%s" % act_values)
        is_available = check_available(act_values)
        if is_available[0]:
            # if self.action_size == 2:
            #     return np.argmax(act_values[0]) + 1  # returns action
            # else:
            #     return np.argmax(act_values[0])  # returns action
            return self.actions[int(np.argmax(act_values[0]))]
        else:
            self.logger.warning("当期状态预测结果无效，选择保持持仓状态。")
            from ibats_common.backend.rl.emulator.market import Action
            return Action.keep

    def get_stochastic_policy(self, inputs):
        """用于模型训练使用，内涵一定几率随机动作"""
        from keras.utils import to_categorical
        self.action_count += 1
        if self.has_update_target_net and np.random.rand() > self.epsilon:
            self.model_predict_count += 1
            # 由于 self.actions[int(np.argmax(act_values[0]))] 以及对上一个动作的 action进行过转化因此不需要再 + 1 了
            # action = inputs[1] + 1 if self.action_size == 2 else inputs[1]
            action = inputs[1]
            act_values = self.model_target.predict(
                x={'state': np.array(inputs[0]), 'flag': to_categorical(action, self.flag_size)})
            if np.any(np.isnan(act_values)):
                self.model_predict_unavailable_count += 1
                self.logger.error(
                    "预测失效=%s。action_count=%4d, model_predict_count=%4d, "
                    "model_predict_unavailable_count=%4d, 当期动作预测失败率=%6.2f%%",
                    act_values, self.action_count, self.model_predict_count, self.model_predict_unavailable_count,
                    self.model_predict_unavailable_rate * 100
                )
                raise ZeroDivisionError("predict error act_values=%s" % act_values)
            is_available = check_available(act_values)
            if is_available[0]:
                action = self.actions[int(np.argmax(act_values[0]))]  # returns action
            else:
                self.model_predict_unavailable_count += 1
                # self.logger.warning(
                #     "当期状态预测结果无效，选择随机策略。action_count=%4d, model_predict_count=%4d, "
                #     "model_predict_unavailable_count=%4d, 当期动作预测失败率=%6.2f%%",
                #     self.action_count, self.model_predict_count, self.model_predict_unavailable_count,
                #     self.model_predict_unavailable_rate * 100
                # )
                action = None
        else:
            action = None

        if action is None:
            if self.last_action is None:
                action = np.random.choice(self.actions)
            else:
                # 随着连续相同动作的数量增加，持续同一动作的概率越来越小
                if np.random.rand() < self.target_avg_holding_rate:
                    action = np.random.choice(self.actions_change_list[self.last_action])
                else:
                    action = self.last_action

        if action == self.last_action:
            self.last_action_same_count += 1
        else:
            self.last_action = action
            self.last_action_same_count = 1

        return self.last_action

    def save_model_weights(self, file_path, ignore_if_unavailable_rate_over=0.4):
        if ignore_if_unavailable_rate_over is not None and \
                ignore_if_unavailable_rate_over > self.model_predict_unavailable_rate:
            return None
        self.model_eval.save_weights(filepath=file_path)
        return file_path

    @property
    def model_predict_unavailable_rate(self):
        return self.model_predict_unavailable_count / self.model_predict_count \
            if self.model_predict_count > 0 else np.nan

    # update target network params
    def update_target_net(self):
        self.model_target.set_weights(self.model_eval.get_weights())

    # update experience replay pool
    def update_cache(self, state, action, reward, next_state, done):
        self.cache_state.append(state)
        # 由于 self.action_size == 2 的情况下 action 只有 0,1 两种状态，而参数action 是 1,2 因此需要 - 1 操作
        if self.action_size == 2:
            self.cache_action.append(action - 1)
        else:
            self.cache_action.append(action)
        self.cache_reward.append(reward)
        self.cache_next_state.append(next_state)
        self.cache_done.append(done)

    # train, update value network params
    def update_value_net(self):
        from keras.utils import to_categorical
        from keras.callbacks import TensorBoard

        # 如果episode数量不到更新所需长度，则讲 state、action、rewards全部保存
        self.tot_update_count += 1
        if self.tot_update_count % self.train_net_period == 0 and len(self.cache_reward_list) > 0:
            # 开始训练网络
            # 找出 cache_*_list 中最长的数据长度
            data_len = np.max([len(_) for _ in self.cache_reward_list])
            rewards_arr = np.full((data_len * self.flag_size, self.action_size), np.nan)


            # 清空 cache_*_list
            self.cache_state_list, self.cache_action_list, self.cache_reward_list, self.cache_next_state_list, \
                self.cache_done_list = [], [], [], [], []
        else:
            # 将数据保存到 cache_*_list
            self.cache_state_list.append(self.cache_state)
            self.cache_action_list.append(self.cache_action)
            self.cache_reward_list.append(self.cache_reward)
            self.cache_next_state_list.append(self.cache_next_state)
            self.cache_done_list.append(self.cache_done)

        if self.tot_update_count % self.update_target_net_period == 0:
            self.update_target_net()
            # 标示网络是否已经被更新
            self.has_update_target_net = True

        # 清空 cache_*
        self.cache_state, self.cache_action, self.cache_reward, self.cache_next_state, self.cache_done = \
            [], [], [], [], []
        # 计算 epsilon
        # 2019-8-21 用衰减的sin函数作为学习曲线
        self.fit_callback.epsilon = self.epsilon = self.epsilon_maker.epsilon_next






        # 以平仓动作为标识，将持仓期间的收益进行反向传递
        # 目的是：每一个动作引发的后续reward也将影响当期记录的最终 reward_tot
        # 以收益率向前叠加的奖励函数
        reward_tot = calc_cum_reward_with_rr(self.cache_reward, self.cum_reward_back_step)
        # 以未来N日calmar为奖励函数（目前发现优化有问题，暂时不清楚原有）
        # reward_tot = calc_cum_reward_with_calmar(self.cache_reward, self.cum_reward_back_step)
        # sum_reward = np.sum(reward_tot)  # 以后可以尝试用此方式优化
        sum_reward = np.sum(self.cache_reward)
        self.cache_list_sum_reward_4_pop_queue.append(sum_reward)
        # 以 reward_tot 为奖励进行训练
        _state = np.concatenate([_[0] for _ in self.cache_state])
        _flag = to_categorical(np.array([_[1] for _ in self.cache_state]), self.flag_size)
        inputs = {'state': _state, 'flag': _flag}
        q_target = self.model_target.predict(x=inputs)
        # 2019-12-24 修复bug “q_target[index, self.cache_action] = reward_tot” 计算结果错误
        # 导致权重数值有误因此无法优化
        # 现改为循环单列赋值方式
        data_len, action_count = q_target.shape
        action_before = np.argmax(q_target, 1)
        actions = np.array(self.cache_action)
        for action in range(action_count):
            matches = actions == action
            q_target[matches, action] = reward_tot[matches]
        action_after = np.argmax(q_target, 1)
        direction_match_rate = np.sum(action_before == action_after) / data_len
        # direction_match_rate 用来标识，初始预测值与在数值修正后的值对比，action 方向性变化率
        self.fit_callback.predict_direction_match_rate = direction_match_rate
        # 对所有无效数据进行惩罚
        is_unavailable = ~check_available(q_target)
        if np.any(is_unavailable) > 0:
            actions = [self.cache_action[_] for _, v in enumerate(is_unavailable) if v]
            self.logger.warning("%d unavailable action: %s value: %s",
                                np.sum(is_unavailable), actions, q_target[is_unavailable, actions])
            q_target[is_unavailable, actions] -= 1.
        # 将训练及进行复制叠加，加入缓存，整理缓存
        self.cache_state_list.append(multiple_data(
            _state[:-self.cum_reward_back_step], self.min_data_len_4_multiple_date))
        self.cache_list_flag.append(multiple_data(
            _flag[:-self.cum_reward_back_step], self.min_data_len_4_multiple_date))
        self.cache_list_q_target.append(multiple_data(
            q_target[:-self.cum_reward_back_step], self.min_data_len_4_multiple_date))
        # 随机删除一组训练样本
        if len(self.cache_list_sum_reward_4_pop_queue) > self.epsilon_memory_size:
            if self.random_drop_cache_rate is not None and np.random.random() < self.random_drop_cache_rate:
                # 随机 drop
                pop_index = np.random.randint(0, self.epsilon_memory_size - 1)
            else:
                # drop 最差方案
                pop_index = int(np.argmin(self.cache_list_sum_reward_4_pop_queue[:self.epsilon_memory_size - 1]))

            self.cache_list_sum_reward_4_pop_queue.pop(pop_index)
            self.cache_state_list.pop(pop_index)
            self.cache_list_flag.pop(pop_index)
            self.cache_list_q_target.pop(pop_index)

        _state = np.concatenate(self.cache_state_list)
        _flag = np.concatenate(self.cache_list_flag)
        _q_target = np.concatenate(self.cache_list_q_target)
        inputs = {'state': _state, 'flag': _flag}
        # 训练并记录损失率，无效率等
        self.fit_callback.model_predict_unavailable_rate = self.model_predict_unavailable_rate
        # 训练模型
        if self.has_update_target_net:
            self.model_eval.fit(
                inputs, _q_target, batch_size=self.batch_size, epochs=self.epochs, verbose=0,
                callbacks=[self.fit_callback],
            )
        else:
            self.model_eval.fit(
                inputs, _q_target, batch_size=self.batch_size, epochs=self.epochs, verbose=0,
                callbacks=[
                    TensorBoard(log_dir=os.path.join(self.tensorboard_log_dir, str(self.tot_update_count))),
                    self.fit_callback],
            )

        return self.acc_loss_lists

    def valid_in_sample(self):
        """利用样本内数据对模型进行验证，返回 loss_dic, valid_rate（样本内数据预测结果有效率）"""
        _state = np.concatenate(self.cache_state_list)
        _flag = np.concatenate(self.cache_list_flag)
        _q_target = np.concatenate(self.cache_list_q_target)
        inputs = {'state': _state, 'flag': _flag}
        losses = self.model_eval.evaluate(inputs, _q_target, verbose=0)
        if len(self.model_eval.metrics_names) == 1:
            loss_dic = {self.model_eval.metrics_names[0]: losses}
        else:
            loss_dic = {name: losses[_] for _, name in enumerate(self.model_eval.metrics_names)}
        _q_pred = self.model_eval.predict(inputs)
        if self.action_size <= 1:
            raise ValueError(f"action_size={self.action_size}, 必须大于1")
        is_available = check_available(_q_pred)
        valid_rate = np.sum(is_available) / len(is_available)

        return loss_dic, valid_rate


def check_available(pred: np.ndarray) -> List[bool]:
    is_available = None
    for idx in range(1, pred.shape[1]):
        if is_available is None:
            is_available = pred[:, idx - 1] != pred[:, idx]
        else:
            is_available |= pred[:, idx - 1] != pred[:, idx]
    return is_available


def calc_cum_reward_with_rr(reward, step, include_curr_day=True, mulitplier=1000, log_value=True):
    """计算 reward 值，将未来 step 步的 reward 以log递减形式向前传导"""
    reward_tot = np.array(reward, dtype='float32')
    tot_len = len(reward_tot)
    data_num, weights = 0, None
    for idx in range(1, tot_len):
        idx_end = min(idx + step, tot_len)
        _data_num = idx_end - idx
        if data_num != _data_num:
            weights = np.logspace(1, _data_num, num=_data_num, base=0.5)
            data_num = _data_num
        if include_curr_day:
            reward_tot[idx - 1] += sum(reward_tot[idx:idx_end] * weights)
        else:
            reward_tot[idx - 1] = sum(reward_tot[idx:idx_end] * weights)

    reward_tot = reward_tot * mulitplier
    if log_value:
        reward_tot[reward_tot > 0] = np.log2(reward_tot[reward_tot > 0] + 1)
        reward_tot[reward_tot < 0] = -np.log2(-reward_tot[reward_tot < 0] + 1)
    return reward_tot


def _test_calc_cum_reward_with_rr():
    """验证 calc_tot_reward 函数"""
    rewards = [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1]
    reward_tot = calc_cum_reward_with_rr(rewards, 3, mulitplier=1, log_value=False)
    print(reward_tot)
    target_reward = np.array([1.875, 1.875, 1.875, 1.75, 1.625, 1.375, 0.875, 1.75, 1.5, 1.125, 0.375, 0.875,
                              1.75, 1.5, 1.])
    assert all(target_reward == reward_tot)


def calc_cum_reward_with_calmar(rewards, win_size=10, threshold=50):
    """计算 reward 值，将未来 step 步的 reward 转化成 calmar，然后以log递减形式向前传导"""
    # 首先将 rewards 转化为 pct，step 返回的 reward 为收益率（带正负值）+1变为 pct
    tot_len = len(rewards)
    reward_tot = (np.array(rewards, dtype='float32') + 1).cumprod()
    # 由于当期状态没有返回日期序列，因此加上从某一日期期按日增加，计算 calmar 值
    reward_s = pd.Series(reward_tot, index=pd.date_range(DATE_BASELINE, periods=tot_len, freq='1D'))
    for num in range(tot_len - 1):
        segment = reward_s[num:num + win_size]
        calmar = segment.calc_calmar_ratio()
        # 对无效数据以及异常数据进行过滤
        if np.inf == calmar:
            if segment[0] < segment[-1]:
                calmar = -threshold
            else:
                calmar = threshold
        elif calmar < -threshold:
            calmar = -threshold
        elif calmar > threshold:
            calmar = threshold

        reward_tot[num] = calmar

    # 最后一天无法计算，标记为0
    reward_tot[tot_len - 1] = 0
    return reward_tot


def _test_calc_cum_reward_with_calmar():
    """验证 calc_tot_reward 函数"""
    rewards = np.sin(np.linspace(1.5, 2 * np.pi, 30)) / 20 + 0.005
    reward_tot = calc_cum_reward_with_calmar(rewards, 20)
    print(reward_tot)
    target_reward = np.array([15.158013, -1.242776, -3.0063176, -3.0086293, -2.8377662, -2.6881402,
                              -2.5780575, -2.5043845, -2.4630232, -2.451551, -2.4515502, -2.4644947,
                              -2.50748, -2.5832598, -2.6970391, -2.8572276, -3.0767963, -3.375636,
                              -3.7847366, -4.353864, -5.1664085, -6.3700013, -8.244485, -11.365294,
                              -17.020409, -28.172022, -47.665047, 50., -50., 0., ])
    assert all([int(x // 0.001) == int(y // 0.001) for x, y in zip(reward_tot, target_reward)])


def _test_show_model():
    from keras.utils import plot_model
    action_size = 2
    agent = Framework(input_shape=[None, 120, 93], action_size=action_size, dueling=True,
                      batch_size=16384)
    file_path = f'model action_size_{action_size}_layer_3.png'
    plot_model(agent.model_eval, to_file=file_path, show_shapes=True)
    from ibats_utils.mess import open_file_with_system_app
    open_file_with_system_app(file_path)


def _test_epsilon_maker():
    import matplotlib.pyplot as plt
    epsilon_maker = EpsilonMaker(sin_step=0.2, epsilon_decay=0.993, epsilon_min=0.05, epsilon_sin_max=0.1)
    epsilon_list = [epsilon_maker.epsilon_next for _ in range(2000)]
    plt.plot(epsilon_list)
    plt.suptitle(
        f'epsilon_decay={epsilon_maker.epsilon_decay:.3f} sin_step={epsilon_maker.sin_step:.2f}, epsilon_min={epsilon_maker.epsilon_min:.2f}')
    plt.show()


def multiple_data(data_list, min_data_len):
    """用于对数据采用二分法复制"""
    data_len, tmp_list = len(data_list), [data_list]
    start_idx = data_len // 2
    while start_idx >= min_data_len:
        tmp_list.append(data_list[-start_idx:])
        start_idx //= 2

    new_data_list = np.concatenate(tmp_list)
    return new_data_list


def _test_multiple_data():
    data_list = [[_, _ * 2] for _ in range(12)]
    new_data_list = multiple_data(data_list, 3)
    # print(new_data_list)
    new_data_list_target = np.array([[0, 0],
                                     [1, 2],
                                     [2, 4],
                                     [3, 6],
                                     [4, 8],
                                     [5, 10],
                                     [6, 12],
                                     [7, 14],
                                     [8, 16],
                                     [9, 18],
                                     [10, 20],
                                     [11, 22],
                                     [6, 12],
                                     [7, 14],
                                     [8, 16],
                                     [9, 18],
                                     [10, 20],
                                     [11, 22],
                                     [9, 18],
                                     [10, 20],
                                     [11, 22],
                                     ])
    assert np.all(new_data_list == new_data_list_target)


if __name__ == '__main__':
    print('import', ffn)
    # _test_calc_tot_reward()
    # _test_show_model()
    _test_calc_cum_reward_with_rr()
    # _test_calc_cum_reward_with_calmar()
    # _test_epsilon_maker()
    # _test_multiple_data()
