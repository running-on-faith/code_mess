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
1）将最近的 max_epsilon_num_4_train 轮训练集作为训练集（而不是仅用当前训练结果作为训练集）
2）训练集超过 max_epsilon_num_4_train 时，将 tot_reward 最低者淘汰，不断保留高 tot_reward 训练集
3) 修改优化器为 Nadam： Nesterov Adam optimizer: Adam本质上像是带有动量项的RMSprop，Nadam就是带有Nesterov 动量的Adam RMSprop
2019-08-21
4) 有 random_drop_best_cache_rate 的几率 随机丢弃 cache 防止最好的用例不断累积造成过度优化
5) EpsilonMaker 提供衰减的sin波动学习曲线
2019-8-23
6）增加了一层网络
"""
import logging

import numpy as np
import tensorflow as tf
from ibats_utils.mess import iter_2_range
from keras import metrics, backend as K
from keras.callbacks import TensorBoard, Callback
from keras.layers import Dense, LSTM, Dropout, Input, concatenate, Lambda
from keras.models import Model
from keras.optimizers import Adam, Nadam
from keras.utils import to_categorical


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


class Framework(object):
    def __init__(self, input_shape=[None, 50, 58, 5], dueling=True, action_size=4, batch_size=512,
                 learning_rate=0.001, tensorboard_log_dir='./log',
                 epochs=1, epsilon_decay=0.9990, sin_step=0.1, epsilon_min=0.05,
                 cum_reward_back_step=5, max_epsilon_num_4_train=5):

        self.input_shape = input_shape
        self.action_size = action_size
        if action_size == 2:
            self.actions = [1, 2]
        else:
            self.actions = list(range(action_size))
        self.cum_reward_back_step = cum_reward_back_step
        self.max_epsilon_num_4_train = max_epsilon_num_4_train
        self.cache_list_state, self.cache_list_flag, self.cache_list_q_target = [], [], []
        self.cache_list_tot_reward = []
        # 2019-07-30
        # [0.1, 0.1, 0.1, 0.7] 有 50% 概率 4步之内保持同一动作
        # [0.15, 0.15, 0.15, 0.55] 有 50% 概率 3步之内保持同一动作
        # 2019-08-08
        # 增加 action_size == 2 多空操作 (1/2)，无空仓
        if action_size == 4:
            self.p = [0.10, 0.2, 0.2, 0.50]
        elif action_size == 3:
            self.p = [0.1, 0.45, 0.45]
        else:
            self.p = None
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.tensorboard_log_dir = tensorboard_log_dir
        self.dueling = dueling
        self.fit_callback = LogFit()

        # cache for experience replay
        # cache for state, action, reward, next_state, done
        self.cache_state, self.cache_action, self.cache_reward, self.cache_next_state, self.cache_done = \
            [], [], [], [], []
        self.weights = None  # 用于 _get_samples 基于权重提取数据
        self.logger = logging.getLogger(str(self.__class__))
        K.clear_session()
        tf.reset_default_graph()

        self.epsilon = 1.0  # exploration rate
        # self.epsilon_min = epsilon_min
        # self.epsilon_decay = epsilon_decay
        self.epsilon_maker = EpsilonMaker(epsilon_decay, sin_step, epsilon_min,
                                          epsilon_sin_max=1 / (cum_reward_back_step * 2))
        self.random_drop_best_cache_rate = 0.01
        self.flag_size = 3
        self.model_eval = self._build_model()
        self.has_logged = False
        self.epochs = epochs

    @property
    def acc_loss_lists(self):
        """return acc_list, loss_list"""
        return self.fit_callback.logs_list

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        input = Input(batch_shape=self.input_shape, name=f'state')
        net = LSTM(self.input_shape[-1] * 2, return_sequences=True, activation='linear')(input)
        net = LSTM(self.input_shape[-1], return_sequences=False, activation='linear')(net)
        net = Dense(self.input_shape[-1] // 2)(net)
        net = Dropout(0.4)(net)
        net = Dense(self.input_shape[-1] // 4)(net)    # 减少一层，降低网络复杂度
        net = Dropout(0.4)(net)
        # net = Dense(self.action_size * 4, activation='relu')(net)
        input2 = Input(batch_shape=[None, self.flag_size], name=f'flag')
        net = concatenate([net, input2])
        net = Dense((self.input_shape[-1] // 4 + self.flag_size) // 2, activation='linear')(net)
        net = Dropout(0.4)(net)
        if self.dueling:
            net = Dense(self.action_size + 1, activation='linear')(net)
            net = Lambda(lambda i: K.expand_dims(i[:, 0], -1) + i[:, 1:] - K.mean(i[:, 1:], keepdims=True),
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

    def get_deterministic_policy_batch(self, inputs, flag_size=None):
        if flag_size is None:
            flag_size = self.flag_size
        act_values = self.model_eval.predict(x={'state': np.array(inputs[0]),
                                                'flag': to_categorical(inputs[1] + 1, flag_size)})
        if self.action_size == 2:
            return np.argmax(act_values, axis=1) + 1
        else:
            return np.argmax(act_values, axis=1)

    def get_deterministic_policy(self, inputs):
        act_values = self.model_eval.predict(x={'state': np.array(inputs[0]),
                                                'flag': to_categorical(inputs[1] + 1, self.flag_size)})
        if self.action_size == 2:
            return np.argmax(act_values[0]) + 1  # returns action
        else:
            return np.argmax(act_values[0])  # returns action

    def get_stochastic_policy(self, inputs):
        if np.random.rand() <= self.epsilon:
            # return random.randrange(self.action_size)
            return np.random.choice(self.actions, p=self.p)
        act_values = self.model_eval.predict(x={'state': np.array(inputs[0]),
                                                'flag': to_categorical(inputs[1] + 1, self.flag_size)})
        if self.action_size == 2:
            return np.argmax(act_values[0]) + 1  # returns action
        else:
            return np.argmax(act_values[0])  # returns action

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

        # 以平仓动作为标识，将持仓期间的收益进行反向传递
        # 目的是：每一个动作引发的后续reward也将影响当期记录的最终 reward_tot
        reward_tot = calc_cum_reward(self.cache_reward, self.cum_reward_back_step)
        tot_reward = np.sum(self.cache_reward)
        if len(self.cache_list_tot_reward) >= self.max_epsilon_num_4_train:
            if np.random.random() < self.random_drop_best_cache_rate:
                # 有一定几率随机drop
                pop_index = np.random.randint(0, self.max_epsilon_num_4_train)
            else:
                pop_index = int(np.argmin(self.cache_list_tot_reward))
            self.cache_list_tot_reward.pop(pop_index)
        else:
            pop_index = None
        self.cache_list_tot_reward.append(tot_reward)
        # 以 reward_tot 为奖励进行训练
        _state = np.concatenate([_[0] for _ in self.cache_state])
        _flag = to_categorical(np.array([_[1] for _ in self.cache_state]) + 1, self.flag_size)
        inputs = {'state': _state,
                  'flag': _flag}
        q_target = self.model_eval.predict(x=inputs)
        index = np.arange(q_target.shape[0])
        q_target[index, self.cache_action] = reward_tot
        # 加入缓存，整理缓存
        self.cache_list_state.append(_state)
        self.cache_list_flag.append(_flag)
        self.cache_list_q_target.append(q_target)
        if pop_index is not None:
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


def calc_cum_reward(reward, step):
    """计算 reward 值，将未来 step 步的 reward 以log递减形式向前传导"""
    reward_tot = np.array(reward, dtype='float32')
    tot_len = len(reward_tot)
    for idx in range(1, tot_len):
        idx_end = min(idx + step, tot_len)
        data_num = idx_end - idx
        weights = np.logspace(1, data_num, num=data_num, base=0.5)
        reward_tot[idx - 1] += sum(reward_tot[idx:idx_end] * weights)

    return reward_tot


def _test_calc_cum_reward():
    """验证 calc_tot_reward 函数"""
    rewards = [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1]
    reward_tot = calc_cum_reward(rewards, 3)
    print(reward_tot)
    target_reward = np.array([1.875, 1.875, 1.875, 1.75, 1.625, 1.375, 0.875, 1.75, 1.5, 1.125, 0.375, 0.875,
                              1.75, 1.5, 1.])
    assert all(target_reward == reward_tot)


def calc_tot_reward(action, reward):
    """计算 reward 值，以每次空仓为界限，将收益以log递减形式向前传导"""
    index_list = [num for num, _ in enumerate(action) if _ == 0]
    if index_list[0] != 0:
        index_list.insert(0, 0)
    if index_list[-1] != (len(action) - 1):
        index_list.append(len(action) - 1)
    range_iter = iter_2_range(index_list, has_right_outer=False, has_left_outer=False)
    reward_tot = np.array(reward, dtype='float32')
    for idx_start, idx_end in range_iter:
        # 下边接 + 1 因为 [idx_start, idx_end)
        for idx in range(idx_start, idx_end):
            data_num = idx_end - idx
            weights = np.logspace(1, data_num, num=data_num, base=0.5)
            reward_tot[idx] += sum(reward_tot[(idx + 1):(idx_end + 1)] * weights)

    return reward_tot


def _test_calc_tot_reward():
    """验证 calc_tot_reward 函数"""
    actions = [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1]
    rewards = [1 for _ in range(len(actions))]
    reward_tot = calc_tot_reward(actions, rewards)
    target_reward = np.array([1.984375, 1.96875, 1.9375, 1.875, 1.75, 1.5,
                              1.9375, 1.875, 1.75, 1.5, 1.5, 1.875,
                              1.75, 1.5, 1.])

    assert all(target_reward == reward_tot)


def _test_show_model():
    from keras.utils import plot_model
    action_size = 2
    agent = Framework(input_shape=[None, 250, 78], action_size=action_size, dueling=True,
                      batch_size=16384)
    file_path = f'model action_size_{action_size}.png'
    plot_model(agent.model_eval, to_file=file_path, show_shapes=True)
    from ibats_utils.mess import open_file_with_system_app
    open_file_with_system_app(file_path)


def _test_epsilon_maker():
    import matplotlib.pyplot as plt
    epsilon_maker = EpsilonMaker(sin_step=0.05, epsilon_decay=0.996, epsilon_min=0.05)
    epsilon_list = [epsilon_maker.epsilon_next for _ in range(2000)]
    plt.plot(epsilon_list)
    plt.suptitle(f'epsilon_decay={epsilon_maker.epsilon_decay:.3f} sin_step={epsilon_maker.sin_step:.2f}, epsilon_min={epsilon_maker.epsilon_min:.2f}')
    plt.show()


if __name__ == '__main__':
    # _test_calc_tot_reward()
    # _test_show_model()
    # _test_calc_cum_reward()
    _test_epsilon_maker()
