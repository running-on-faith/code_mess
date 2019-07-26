import logging
import random

import numpy as np
import tensorflow as tf
from ibats_utils.mess import iter_2_range
from keras import metrics, backend as K
from keras.callbacks import TensorBoard, Callback
from keras.layers import Dense, LSTM, Dropout, Input, concatenate, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical

_EPSILON = 1e-6  # avoid nan


class LogFit(Callback):

    def __init__(self):
        super().__init__()
        self.loss_list = []
        self.acc_list = []
        self.logger = logging.getLogger(str(self.__class__))

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            # self.logger.debug('%s', logs)
            self.loss_list.append(logs['loss'] if 'loss' in logs else np.nan)
            self.acc_list.append(logs['acc'] if 'acc' in logs else np.nan)


class Framework(object):
    def __init__(self, memory_size=2048, input_shape=[None, 50, 58, 5], dueling=False, action_size=4, batch_size=512,
                 gamma=0.95, learning_rate=0.001, tensorboard_log_dir='./log'):

        self.memory_size = memory_size
        self.input_shape = input_shape
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
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
        self.epsilon_min = 0.3
        self.epsilon_decay = 0.9998
        self.flag_size = 3
        self.model_eval = self._build_model()
        self.has_logged = False

    @property
    def acc_loss_lists(self):
        """return acc_list, loss_list"""
        return self.fit_callback.acc_list, self.fit_callback.loss_list

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        input = Input(batch_shape=self.input_shape, name=f'state')
        net = LSTM(self.input_shape[-1] * 2, return_sequences=True, activation='relu')(input)
        net = LSTM(self.input_shape[-1], return_sequences=False, activation='relu')(net)
        net = Dense(self.input_shape[-1] // 2)(net)
        net = Dropout(0.4)(net)
        # net = Dense(self.input_shape[-1])(net)    # 减少一层，降低网络复杂度
        # net = Dropout(0.4)(net)
        # net = Dense(self.action_size * 4, activation='relu')(net)
        input2 = Input(batch_shape=[None, self.flag_size], name=f'flag')
        net = concatenate([net, input2])
        net = Dense(self.action_size * 2 + self.flag_size, activation='relu')(net)
        net = Dropout(0.4)(net)
        if self.dueling:
            net = Dense(self.action_size + 1, activation='linear')(net)
            net = Lambda(lambda i: K.expand_dims(i[:, 0], -1) + i[:, 1:] - K.mean(i[:, 1:], keepdims=True),
                         output_shape=(self.action_size,))(net)
        else:
            net = Dense(self.action_size, activation='relu')(net)

        model = Model(inputs=[input, input2], outputs=net)
        model.summary()
        model.compile(Adam(self.learning_rate), loss=self._huber_loss,
                      metrics=[metrics.mae, metrics.categorical_accuracy])
        return model

    def get_deterministic_policy_batch(self, inputs):
        act_values = self.model_eval.predict(x={'state': np.array(inputs[0]),
                                                'flag': to_categorical(inputs[1] + 1, self.flag_size)})
        return np.argmax(act_values, axis=1)

    def get_deterministic_policy(self, inputs):
        act_values = self.model_eval.predict(x={'state': np.array(inputs[0]),
                                                'flag': to_categorical(inputs[1] + 1, self.flag_size)})
        return np.argmax(act_values[0])  # returns action

    def get_stochastic_policy(self, inputs):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model_eval.predict(x={'state': np.array(inputs[0]),
                                                'flag': to_categorical(inputs[1] + 1, self.flag_size)})
        return np.argmax(act_values[0])  # returns action

    # update experience replay pool
    def update_cache(self, state, action, reward, next_state, done):
        self.cache_state.append(state)
        self.cache_action.append(action)
        self.cache_reward.append(reward)
        self.cache_next_state.append(next_state)
        self.cache_done.append(done)

    # train, update value network params
    def update_value_net(self):

        # 以平仓动作为标识，将持仓期间的收益进行反向传递
        # 目的是：每一个动作引发的后续reward也将影响当期记录的最终 reward_tot
        reward_tot = calc_tot_reward(self.cache_action, self.cache_reward)
        # 以 reward_tot 为奖励进行训练

        inputs = {'state': np.concatenate([_[0] for _ in self.cache_state]),
                  'flag': to_categorical(np.array([_[1] for _ in self.cache_state]) + 1, self.flag_size)}
        q_target = self.model_eval.predict(x=inputs)
        index = np.arange(q_target.shape[0])
        q_target[index, self.cache_action] = reward_tot

        if self.has_logged:
            self.model_eval.fit(inputs, q_target, batch_size=self.batch_size, epochs=1,
                                verbose=0, callbacks=[self.fit_callback],
                                )
        else:
            self.model_eval.fit(inputs, q_target, batch_size=self.batch_size, epochs=1,
                                verbose=0, callbacks=[TensorBoard(log_dir='./tensorboard_log'), self.fit_callback],
                                )
            self.has_logged = True
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return self.acc_loss_lists


def calc_tot_reward(action, reward):
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
    agent = Framework(input_shape=[None, 250, 78], action_size=3, dueling=True,
                      gamma=0.3, batch_size=512, memory_size=100000)
    file_path = 'model.png'
    plot_model(agent.model_eval, to_file=file_path, show_shapes=True)
    from ibats_utils.mess import open_file_with_system_app
    open_file_with_system_app(file_path)


if __name__ == '__main__':
    # _test_calc_tot_reward()
    _test_show_model()
