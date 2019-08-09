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
        self.logs_list = []
        self.epsilon = 1.
        self.logger = logging.getLogger(str(self.__class__))

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            # self.logger.debug('%s', logs)
            # self.loss_list.append(logs['loss'] if 'loss' in logs else np.nan)
            # self.acc_list.append(logs['acc'] if 'acc' in logs else np.nan)
            # self.acc_list.append(logs['categorical_accuracy'] if 'categorical_accuracy' in logs else np.nan)
            # self.acc_list.append(logs['mean_absolute_error'] if 'mean_absolute_error' in logs else np.nan)
            logs['epsilon'] = self.epsilon
            self.logs_list.append(logs)


class Framework(object):
    def __init__(self, input_shape=[None, 50, 58, 5], dueling=False, action_size=4, batch_size=512,
                 gamma=0.95, learning_rate=0.001, tensorboard_log_dir='./log',
                 epochs=1, epsilon_decay=0.9990, epsilon_min=0.2):

        self.input_shape = input_shape
        self.action_size = action_size
        if action_size == 2:
            self.actions = [1, 2]
        else:
            self.actions = list(range(action_size))
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
            self.p = [1/action_size for _ in range(action_size)]
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
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.flag_size = 3
        self.model_eval = self._build_model()
        self.has_logged = False
        self.epochs = epochs

    @property
    def acc_loss_lists(self):
        """return acc_list, loss_list"""
        return self.fit_callback.logs_list

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        input = Input(batch_shape=self.input_shape, name=f'state')
        net = LSTM(self.input_shape[-1] * 2, return_sequences=True, activation='linear')(input)
        net = LSTM(self.input_shape[-1], return_sequences=False, activation='linear')(net)
        net = Dense(self.input_shape[-1] // 2)(net)
        net = Dropout(0.4)(net)
        # net = Dense(self.input_shape[-1])(net)    # 减少一层，降低网络复杂度
        # net = Dropout(0.4)(net)
        # net = Dense(self.action_size * 4, activation='relu')(net)
        input2 = Input(batch_shape=[None, self.flag_size], name=f'flag')
        net = concatenate([net, input2])
        net = Dense((self.input_shape[-1] // 2 + self.flag_size) // 2, activation='linear')(net)
        net = Dropout(0.4)(net)
        if self.dueling:
            net = Dense(self.action_size + 1, activation='linear')(net)
            net = Lambda(lambda i: K.expand_dims(i[:, 0], -1) + i[:, 1:] - K.mean(i[:, 1:], keepdims=True),
                         output_shape=(self.action_size,))(net)
        else:
            net = Dense(self.action_size, activation='linear')(net)

        model = Model(inputs=[input, input2], outputs=net)
        model.compile(Adam(self.learning_rate), loss=self._huber_loss,
                      metrics=[metrics.mae, metrics.categorical_accuracy]
                      )
        # model.summary()
        return model

    def get_deterministic_policy_batch(self, inputs):
        act_values = self.model_eval.predict(x={'state': np.array(inputs[0]),
                                                'flag': to_categorical(inputs[1] + 1, self.flag_size)})
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
        reward_tot = calc_tot_reward(self.cache_action, self.cache_reward)
        # 以 reward_tot 为奖励进行训练

        inputs = {'state': np.concatenate([_[0] for _ in self.cache_state]),
                  'flag': to_categorical(np.array([_[1] for _ in self.cache_state]) + 1, self.flag_size)}
        q_target = self.model_eval.predict(x=inputs)
        index = np.arange(q_target.shape[0])
        q_target[index, self.cache_action] = reward_tot

        if self.has_logged:
            self.model_eval.fit(inputs, q_target, batch_size=self.batch_size, epochs=self.epochs,
                                verbose=0, callbacks=[self.fit_callback],
                                )
        else:
            self.model_eval.fit(inputs, q_target, batch_size=self.batch_size, epochs=self.epochs,
                                verbose=0, callbacks=[TensorBoard(log_dir='./tensorboard_log'), self.fit_callback],
                                )
            self.has_logged = True
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.fit_callback.epsilon = self.epsilon

        self.cache_state, self.cache_action, self.cache_reward, self.cache_next_state, self.cache_done = \
            [], [], [], [], []

        return self.acc_loss_lists


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
                      gamma=0.3, batch_size=512)
    file_path = f'model action_size_{action_size}.png'
    plot_model(agent.model_eval, to_file=file_path, show_shapes=True)
    from ibats_utils.mess import open_file_with_system_app
    open_file_with_system_app(file_path)


if __name__ == '__main__':
    # _test_calc_tot_reward()
    _test_show_model()
