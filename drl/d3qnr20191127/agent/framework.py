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
"""
import logging

import ffn
import numpy as np
import pandas as pd

DATE_BASELINE = pd.to_datetime('2018-01-01')


class EpsilonMaker:
    def __init__(self, epsilon_decay=0.996, sin_step=0.05, epsilon_min=0.05, epsilon_sin_max=0.2):
        self.sin_step = sin_step
        self.sin_step_tot = 0
        self.epsilon = self.epsilon_init = 1.0  # exploration rate
        self.epsilon_sin_max = epsilon_sin_max
        self.epsilon_min = epsilon_min
        self.epsilon_down = self.epsilon  # self.epsilon_down *= self.epsilon_decay
        self.epsilon_sin = 0
        self.sin_height = self.epsilon_sin_max - self.epsilon_min
        self.epsilon_decay = epsilon_decay

    @property
    def epsilon_next(self):
        if self.epsilon_down > self.epsilon_min:
            self.epsilon_down *= self.epsilon_decay
        self.epsilon_sin = (np.sin(self.sin_step_tot * self.sin_step) + 1) / 2 * self.sin_height
        self.sin_step_tot += 1
        self.epsilon = self.epsilon_down + self.epsilon_sin
        if self.epsilon > self.epsilon_init:
            self.epsilon = self.epsilon_init
        elif self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

        return self.epsilon


def build_model_8_layers(input_shape, flag_size, action_size, reg_params=[1e-7, 1e-7, None], learning_rate=0.001,
                         dueling=True, is_classification=False):
    import tensorflow as tf
    from keras.layers import Dense, LSTM, Dropout, Input, concatenate, Lambda, Activation
    from keras.models import Model
    from keras import metrics, backend as K
    from keras.optimizers import Nadam
    from keras.regularizers import l2, l1_l2
    # Neural Net for Deep-Q learning Model
    input = Input(batch_shape=input_shape, name=f'state')
    # 2019-11-27 增加对 LSTM 层的正则化
    # 根据 《使用权重症则化较少模型过拟合》，经验上，LSTM 正则化 10^-6
    # 使用 L1L2 混合正则项（又称：Elastic Net）
    recurrent_regularizer = l1_l2(l1=reg_params[0], l2=reg_params[1]) \
        if reg_params[0] is not None and reg_params[1] is not None else None
    kernel_regularizer = l2(reg_params[2]) if reg_params[2] is not None else None
    net = LSTM(
        input_shape[-1] * 2,
        recurrent_regularizer=recurrent_regularizer,
        kernel_regularizer=kernel_regularizer,
        dropout=0.3
    )(input)
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
        net = Lambda(lambda i: K.expand_dims(i[:, 0], -1) + i[:, 1:] - K.mean(i[:, 1:], keepdims=True),
                     output_shape=(action_size,))(net)
    else:
        net = Dense(action_size, activation='linear')(net)

    if is_classification:
        net = Activation('softmax')(net)

    model = Model(inputs=[input, input2], outputs=net)

    def _huber_loss(y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

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
                      metrics=[metrics.mae, metrics.mean_squared_logarithmic_error]
                      )
    # model.summary()
    return model


def build_model_5_layers(input_shape, flag_size, action_size, reg_params=[1e-7, 1e-7, None], learning_rate=0.001,
                         dueling=True, is_classification=False):
    import tensorflow as tf
    from keras.layers import Dense, LSTM, Dropout, Input, concatenate, Lambda, Activation
    from keras.models import Model
    from keras import metrics, backend as K
    from keras.optimizers import Nadam
    from keras.regularizers import l2, l1_l2
    # Neural Net for Deep-Q learning Model
    input = Input(batch_shape=input_shape, name=f'state')
    # 2019-11-27 增加对 LSTM 层的正则化
    # 根据 《使用权重症则化较少模型过拟合》，经验上，LSTM 正则化 10^-6
    # 使用 L1L2 混合正则项（又称：Elastic Net）
    # TODO: 参数尚未优化
    recurrent_regularizer = l1_l2(l1=reg_params[0], l2=reg_params[1]) \
        if reg_params[0] is not None and reg_params[1] is not None else None
    kernel_regularizer = l2(reg_params[2]) if reg_params[2] is not None else None
    input_size = input_shape[-1]
    net = LSTM(
        input_size * 2,
        recurrent_regularizer=recurrent_regularizer,
        kernel_regularizer=kernel_regularizer,
        dropout=0.3
    )(input)
    net = Dense(int(input_size / 2))(net)
    net = Dropout(0.4)(net)
    input2 = Input(batch_shape=[None, flag_size], name=f'flag')
    net = concatenate([net, input2])
    net = Dense(int((input_size / 2 + flag_size) / 4))(net)
    net = Dropout(0.4)(net)
    if dueling:
        net = Dense(action_size + 1, activation='relu')(net)
        net = Lambda(lambda i: K.expand_dims(i[:, 0], -1) + i[:, 1:] - K.mean(i[:, 1:], keepdims=True),
                     output_shape=(action_size,))(net)
    else:
        net = Dense(action_size, activation='linear')(net)

    if is_classification:
        net = Activation('softmax')(net)

    model = Model(inputs=[input, input2], outputs=net)

    def _huber_loss(y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

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
                      metrics=[metrics.mae, metrics.mean_squared_logarithmic_error]
                      )
    # model.summary()
    return model


def build_model_4_layers(input_shape, flag_size, action_size, reg_params=[1e-7, 1e-7, None], learning_rate=0.001,
                         dueling=True, is_classification=False):
    import tensorflow as tf
    from keras.layers import Dense, LSTM, Dropout, Input, concatenate, Lambda, Activation
    from keras.models import Model
    from keras import metrics, backend as K
    from keras.optimizers import Nadam
    from keras.regularizers import l2, l1_l2
    # Neural Net for Deep-Q learning Model
    input = Input(batch_shape=input_shape, name=f'state')
    # 2019-11-27 增加对 LSTM 层的正则化
    # 根据 《使用权重症则化较少模型过拟合》，经验上，LSTM 正则化 10^-6
    # 使用 L1L2 混合正则项（又称：Elastic Net）
    # TODO: 参数尚未优化
    recurrent_regularizer = l1_l2(l1=reg_params[0], l2=reg_params[1]) \
        if reg_params[0] is not None and reg_params[1] is not None else None
    kernel_regularizer = l2(reg_params[2]) if reg_params[2] is not None else None
    input_size = input_shape[-1]
    net = LSTM(
        input_size * 2,
        recurrent_regularizer=recurrent_regularizer,
        kernel_regularizer=kernel_regularizer,
        dropout=0.3
    )(input)
    input2 = Input(batch_shape=[None, flag_size], name=f'flag')
    net = concatenate([net, input2])
    net = Dense(int((input_size + flag_size) / 4))(net)
    net = Dropout(0.4)(net)
    if dueling:
        net = Dense(action_size + 1, activation='relu')(net)
        net = Lambda(lambda i: K.expand_dims(i[:, 0], -1) + i[:, 1:] - K.mean(i[:, 1:], keepdims=True),
                     output_shape=(action_size,))(net)
    else:
        net = Dense(action_size, activation='linear')(net)

    if is_classification:
        net = Activation('softmax')(net)

    model = Model(inputs=[input, input2], outputs=net)

    def _huber_loss(y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

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
                      metrics=[metrics.mae, metrics.mean_squared_logarithmic_error]
                      )
    # model.summary()
    return model


def build_model_3_layers(input_shape, flag_size, action_size, reg_params=[1e-7, 1e-7, None],
                         learning_rate=0.001, dueling=True, is_classification=False):
    import tensorflow as tf
    from keras.layers import Dense, LSTM, Input, concatenate, Lambda, Activation
    from keras.models import Model
    from keras import metrics, backend as K
    from keras.optimizers import Nadam
    from keras.regularizers import l2, l1_l2
    # Neural Net for Deep-Q learning Model
    input = Input(batch_shape=input_shape, name=f'state')
    # 2019-11-27 增加对 LSTM 层的正则化
    # 根据 《使用权重症则化较少模型过拟合》，经验上，LSTM 正则化 10^-6
    # 使用 L1L2 混合正则项（又称：Elastic Net）
    # TODO: 参数尚未优化
    recurrent_regularizer = l1_l2(l1=reg_params[0], l2=reg_params[1]) \
        if reg_params[0] is not None and reg_params[1] is not None else None
    kernel_regularizer = l2(reg_params[2]) if reg_params[2] is not None else None
    input_size = input_shape[-1]
    net = LSTM(
        input_size * 2,
        recurrent_regularizer=recurrent_regularizer,
        kernel_regularizer=kernel_regularizer,
        dropout=0.3
    )(input)
    input2 = Input(batch_shape=[None, flag_size], name=f'flag')
    net = concatenate([net, input2])
    if dueling:
        net = Dense(action_size + 1, activation='relu')(net)
        net = Lambda(lambda i: K.expand_dims(i[:, 0], -1) + i[:, 1:] - K.mean(i[:, 1:], keepdims=True),
                     output_shape=(action_size,))(net)
    else:
        net = Dense(action_size, activation='linear')(net)

    if is_classification:
        net = Activation('softmax')(net)

    model = Model(inputs=[input, input2], outputs=net)

    def _huber_loss(y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

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
                      metrics=[metrics.mae, metrics.mean_squared_logarithmic_error]
                      )
    # model.summary()
    return model


class Framework(object):
    def __init__(self, input_shape=[None, 60, 93], dueling=True, action_size=4, batch_size=512,
                 learning_rate=0.001, tensorboard_log_dir='./log',
                 epochs=1, epsilon_decay=0.9990, sin_step=0.1, epsilon_min=0.05, update_target_net_period=20,
                 cum_reward_back_step=10, epsilon_memory_size=20, keep_last_action=0.9057,
                 min_data_len_4_multiple_date=30, random_drop_best_cache_rate=0.01,
                 reg_params=[1e-7, 1e-7, 1e-3]):
        import tensorflow as tf
        from keras import backend as K
        from keras.callbacks import Callback

        class LogFit(Callback):

            def __init__(self):
                super().__init__()
                self.logs_list = []
                self.epsilon = 1.
                self.logger = logging.getLogger(str(self.__class__))

            def on_epoch_end(self, epoch, logs=None):
                if logs is not None and len(logs) > 0:
                    # self.logger.debug('%s', logs)
                    # self.loss_list.append(logs['loss'] if 'loss' in logs else np.nan)
                    # self.acc_list.append(logs['acc'] if 'acc' in logs else np.nan)
                    # self.acc_list.append(logs['categorical_accuracy'] if 'categorical_accuracy' in logs else np.nan)
                    # self.acc_list.append(logs['mean_absolute_error'] if 'mean_absolute_error' in logs else np.nan)
                    logs['epsilon'] = self.epsilon
                    self.logs_list.append(logs)

        self.input_shape = [None if num == 0 else _ for num, _ in enumerate(input_shape)]
        self.action_size = action_size
        if action_size == 2:
            self.actions = [1, 2]
        else:
            self.actions = list(range(action_size))
        self.last_action = None
        self.last_action_same_count = 0
        # 延续上一执行动作的概率
        # keep_last_action=0.84     math.pow(0.5, 0.25) = 0.84089
        # keep_last_action=0.87     math.pow(0.5, 0.20) = 0.87055
        # keep_last_action=0.8816   math.pow(0.5, 0.1818) = 0.8816
        # keep_last_action=0.9057   math.pow(0.5, 1/7) = 0.9057
        # keep_last_action=0.9170   math.pow(0.5, 0.125) = 0.9170
        self.keep_last_action = keep_last_action
        self.cum_reward_back_step = cum_reward_back_step
        self.epsilon_memory_size = epsilon_memory_size
        self.cache_list_state, self.cache_list_flag, self.cache_list_q_target = [], [], []
        self.cache_list_tot_reward = []
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.tensorboard_log_dir = tensorboard_log_dir
        self.dueling = dueling
        self.fit_callback = LogFit()
        # cache for experience replay
        # cache for state, action, reward, next_state, done
        self.cache_state, self.cache_action, self.cache_reward, self.cache_next_state, self.cache_done = \
            [], [], [], [], []
        self.logger = logging.getLogger(str(self.__class__))
        K.clear_session()
        tf.reset_default_graph()

        self.epsilon = 1.0  # exploration rate
        # self.epsilon_min = epsilon_min
        # self.epsilon_decay = epsilon_decay
        self.epsilon_maker = EpsilonMaker(epsilon_decay, sin_step, epsilon_min,
                                          epsilon_sin_max=0.05)
        self.random_drop_best_cache_rate = random_drop_best_cache_rate
        self.flag_size = 3
        self.reg_params = reg_params
        self.model_eval = self._build_model()
        self.model_target = self._build_model()
        self.has_logged = False
        self.epochs = epochs
        self.update_target_net_period = update_target_net_period
        self.tot_update_count = 0
        self.min_data_len_4_multiple_date = min_data_len_4_multiple_date

    @property
    def acc_loss_lists(self):
        """return acc_list, loss_list"""
        return self.fit_callback.logs_list

    def _build_model(self):
        return build_model_3_layers(
            input_shape=self.input_shape, flag_size=self.flag_size, action_size=self.action_size,
            reg_params=self.reg_params, learning_rate=self.learning_rate, dueling=self.dueling)

    def _build_model_20191127(self):
        import tensorflow as tf
        from keras.layers import Dense, LSTM, Dropout, Input, concatenate, Lambda
        from keras.models import Model
        from keras import metrics, backend as K
        from keras.optimizers import Nadam
        from keras.regularizers import l2, l1_l2
        # Neural Net for Deep-Q learning Model
        input = Input(batch_shape=self.input_shape, name=f'state')
        # 2019-11-27 增加对 LSTM 层的正则化
        # 根据 《使用权重症则化较少模型过拟合》，经验上，LSTM 正则化 10^-6
        # 使用 L1L2 混合正则项（又称：Elastic Net）
        # TODO: 参数尚未优化
        recurrent_regularizer = l1_l2(l1=self.reg_params[0], l2=self.reg_params[1]) \
            if self.reg_params[0] is not None and self.reg_params[1] is not None else None
        kernel_regularizer = l2(self.reg_params[2]) if self.reg_params[2] is not None else None
        net = LSTM(
            self.input_shape[-1] * 2,
            recurrent_regularizer=recurrent_regularizer,
            kernel_regularizer=kernel_regularizer)(input)
        net = Dense(int(self.input_shape[-1] / 1.5))(net)
        net = Dropout(0.4)(net)
        input2 = Input(batch_shape=[None, self.flag_size], name=f'flag')
        net = concatenate([net, input2])
        net = Dense((int(self.input_shape[-1] / 1.5) + self.flag_size) // 3, activation='linear')(net)
        net = Dropout(0.4)(net)
        # net = Dense(self.action_size * 4, activation='relu')(net)
        if self.dueling:
            net = Dense(self.action_size + 2, activation='linear')(net)
            net = Lambda(lambda i: K.expand_dims(i[:, 0], -1) + i[:, 2:] - K.mean(i[:, 2:], keepdims=True),
                         output_shape=(self.action_size,))(net)
        else:
            net = Dense(self.action_size, activation='linear')(net)

        model = Model(inputs=[input, input2], outputs=net)

        def _huber_loss(y_true, y_pred, clip_delta=1.0):
            error = y_true - y_pred
            cond = K.abs(error) <= clip_delta

            squared_loss = 0.5 * K.square(error)
            quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

            return K.mean(tf.where(cond, squared_loss, quadratic_loss))

        model.compile(Nadam(self.learning_rate), loss=_huber_loss,
                      metrics=[metrics.mae, metrics.mean_squared_error, metrics.categorical_accuracy]
                      )
        # model.summary()
        return model

    def get_deterministic_policy(self, inputs):
        from keras.utils import to_categorical
        action = inputs[1] + 1 if self.action_size == 2 else inputs[1]
        # self.logger.debug('flag.shape=%s, flag=%s', np.array(inputs[0]).shape, to_categorical(action, self.flag_size))
        act_values = self.model_eval.predict(x={'state': np.array(inputs[0]),
                                                'flag': to_categorical(action, self.flag_size)})
        if np.any(np.isnan(act_values)):
            self.logger.error("predict error act_values=%s", act_values)
            raise ZeroDivisionError("predict error act_values=%s" % act_values)
        if self.action_size == 2:
            return np.argmax(act_values[0]) + 1  # returns action
        else:
            return np.argmax(act_values[0])  # returns action

    def get_stochastic_policy(self, inputs):
        from keras.utils import to_categorical
        if np.random.rand() <= self.epsilon:
            if self.last_action is None:
                action = np.random.choice(self.actions)
            else:
                # 随着连续相同动作的数量增加，持续同一动作的概率越来越小
                if np.random.rand() > self.keep_last_action ** self.last_action_same_count:
                    action = np.random.choice(self.actions)
                else:
                    action = self.last_action
        else:
            act_values = self.model_eval.predict(x={'state': np.array(inputs[0]),
                                                    'flag': to_categorical(inputs[1] + 1, self.flag_size)})
            if np.any(np.isnan(act_values)):
                self.logger.error("predict error act_values=%s", act_values)
                raise ZeroDivisionError("predict error act_values=%s" % act_values)
            action = self.actions[int(np.argmax(act_values[0]))]  # returns action

        if action == self.last_action:
            self.last_action_same_count += 1
        else:
            self.last_action = action
            self.last_action_same_count = 1
        return self.last_action

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

        self.tot_update_count += 1
        if self.tot_update_count % self.update_target_net_period == 0:
            self.update_target_net()

        # 以平仓动作为标识，将持仓期间的收益进行反向传递
        # 目的是：每一个动作引发的后续reward也将影响当期记录的最终 reward_tot
        # 以收益率向前叠加的奖励函数
        reward_tot = calc_cum_reward_with_rr(self.cache_reward, self.cum_reward_back_step)
        # 以未来N日calmar为奖励函数（目前发现优化有问题，暂时不清楚原有）
        # reward_tot = calc_cum_reward_with_calmar(self.cache_reward, self.cum_reward_back_step)
        tot_reward = np.sum(self.cache_reward)
        self.cache_list_tot_reward.append(tot_reward)
        # 以 reward_tot 为奖励进行训练
        _state = np.concatenate([_[0] for _ in self.cache_state])
        _flag = to_categorical(np.array([_[1] for _ in self.cache_state]) + 1, self.flag_size)
        inputs = {'state': _state,
                  'flag': _flag}
        q_target = self.model_target.predict(x=inputs)
        index = np.arange(q_target.shape[0])
        q_target[index, self.cache_action] = reward_tot
        # 将训练及进行复制叠加
        # 加入缓存，整理缓存
        self.cache_list_state.append(multiple_data(
            _state[:-self.cum_reward_back_step], self.min_data_len_4_multiple_date))
        self.cache_list_flag.append(multiple_data(
            _flag[:-self.cum_reward_back_step], self.min_data_len_4_multiple_date))
        self.cache_list_q_target.append(multiple_data(
            q_target[:-self.cum_reward_back_step], self.min_data_len_4_multiple_date))
        # 随机删除一组训练样本
        if len(self.cache_list_tot_reward) >= self.epsilon_memory_size:
            if np.random.random() < self.random_drop_best_cache_rate:
                # 有一定几率随机drop
                pop_index = np.random.randint(0, self.epsilon_memory_size - 1)
            else:
                pop_index = int(np.argmin(self.cache_list_tot_reward))

            self.cache_list_tot_reward.pop(pop_index)
            self.cache_list_state.pop(pop_index)
            self.cache_list_flag.pop(pop_index)
            self.cache_list_q_target.pop(pop_index)

        _state = np.concatenate(self.cache_list_state)
        _flag = np.concatenate(self.cache_list_flag)
        _q_target = np.concatenate(self.cache_list_q_target)
        inputs = {'state': _state, 'flag': _flag}
        if self.has_logged:
            self.model_eval.fit(inputs, _q_target, batch_size=self.batch_size, epochs=self.epochs,
                                verbose=0, callbacks=[self.fit_callback],
                                )
        else:
            self.model_eval.fit(inputs, _q_target, batch_size=self.batch_size, epochs=self.epochs,
                                verbose=0, callbacks=[TensorBoard(log_dir='./tensorboard_log'), self.fit_callback],
                                )
            self.has_logged = True

        # 计算 epsilon
        # 2019-8-21 用衰减的sin函数作为学习曲线
        self.fit_callback.epsilon = self.epsilon = self.epsilon_maker.epsilon_next
        # 原来的 epsilon 计算函数
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
        #     self.fit_callback.epsilon = self.epsilon

        self.cache_state, self.cache_action, self.cache_reward, self.cache_next_state, self.cache_done = \
            [], [], [], [], []

        return self.acc_loss_lists

    def valid_in_sample(self):
        """利用样本内数据对模型进行验证，返回 loss_dic, valid_rate（样本内数据预测结果有效率）"""
        _state = np.concatenate(self.cache_list_state)
        _flag = np.concatenate(self.cache_list_flag)
        _q_target = np.concatenate(self.cache_list_q_target)
        inputs = {'state': _state, 'flag': _flag}
        losses = self.model_eval.evaluate(inputs, _q_target, verbose=0)
        loss_dic = {name: losses[_] for _, name in enumerate(self.model_eval.metrics_names)}
        _q_pred = self.model_eval.predict(inputs)
        is_valid = None
        for idx in range(1, self.action_size):
            if is_valid is None:
                is_valid = _q_pred[:, idx - 1] != _q_pred[:, idx]
            else:
                is_valid |= _q_pred[:, idx - 1] != _q_pred[:, idx]

        valid_rate = np.sum(is_valid) / len(is_valid)
        return loss_dic, valid_rate


def calc_cum_reward_with_rr(reward, step, include_curr_day=True, mulitplier=1000):
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

    return reward_tot * mulitplier


def _test_calc_cum_reward_with_rr():
    """验证 calc_tot_reward 函数"""
    rewards = [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1]
    reward_tot = calc_cum_reward_with_rr(rewards, 3)
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
    epsilon_maker = EpsilonMaker(sin_step=0.05, epsilon_decay=0.996, epsilon_min=0.05)
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
    _test_show_model()
    # _test_calc_cum_reward_with_rr()
    # _test_calc_cum_reward_with_calmar()
    # _test_epsilon_maker()
    # _test_multiple_data()
