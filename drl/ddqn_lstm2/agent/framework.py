import logging
import random
from collections import deque

import numpy as np
import tensorflow as tf
from ibats_utils.mess import sample_weighted
from keras import backend as K
from keras.callbacks import TensorBoard, Callback
from keras.layers import Dense, LSTM, Dropout, Input, concatenate, Lambda, Reshape
from keras.models import Model
from keras.optimizers import Adam

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
        self.cache = deque(maxlen=memory_size)
        self.weights = None  # 用于 _get_samples 基于权重提取数据
        self.logger = logging.getLogger(str(self.__class__))
        K.clear_session()
        tf.reset_default_graph()

        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.3
        self.epsilon_decay = 0.9998
        self.model_eval = self._build_model_eval()
        self.model_target = self._build_model_target()
        self.update_target_net()
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

    def _build_model_eval(self):
        # Neural Net for Deep-Q learning Model
        input = Input(batch_shape=self.input_shape, name=f'state_eval')
        lstm = LSTM(self.input_shape[-1] * 2, return_sequences=True, activation='relu')(input)
        lstm = LSTM(self.input_shape[-1], return_sequences=False, activation='relu')(lstm)
        d1 = Dense(self.input_shape[-1] * 2)(lstm)
        dr1 = Dropout(0.4)(d1)
        d2 = Dense(self.input_shape[-1])(dr1)
        dr2 = Dropout(0.4)(d2)
        d3 = Dense(self.action_size * 2, activation='relu')(dr2)
        input2 = Input(batch_shape=[None, 1], name=f'flag_eval')
        # flag_hot = Reshape((3,))(K.one_hot(input2, 3))
        concat = concatenate([d3, input2])
        if self.dueling:
            d41 = Dense(self.action_size + 1, activation='linear')(concat)
            d4 = Lambda(lambda i: K.expand_dims(i[:, 0], -1) + i[:, 1:] - K.mean(i[:, 1:], keepdims=True),
                        output_shape=(self.action_size,))(d41)
        else:
            d4 = Dense(self.action_size, activation='relu')(concat)

        model = Model(inputs=[input, input2], outputs=d4)
        model.summary()
        model.compile(Adam(self.learning_rate), loss=self._huber_loss, metrics=['accuracy'])
        return model

    def _build_model_target(self):
        """由于 log 输出graph发现 Dropout 相互之间总是有连线，于是选择将两个图完全独立制作"""
        # Neural Net for Deep-Q learning Model
        input = Input(batch_shape=self.input_shape, name=f'state_target')
        lstm = LSTM(self.input_shape[-1] * 2, return_sequences=True, activation='relu')(input)
        lstm = LSTM(self.input_shape[-1], return_sequences=False, activation='relu')(lstm)
        d1 = Dense(self.input_shape[-1] * 2)(lstm)
        dr1 = Dropout(0.4)(d1)
        d2 = Dense(self.input_shape[-1])(dr1)
        dr2 = Dropout(0.4)(d2)
        d3 = Dense(self.action_size * 2, activation='relu')(dr2)
        input2 = Input(batch_shape=[None, 1], name=f'flag_target')
        # flag_hot = Reshape((3,))(K.one_hot(input2, 3))
        concat = concatenate([d3, input2])
        if self.dueling:
            d41 = Dense(self.action_size + 1, activation='linear')(concat)
            d4 = Lambda(lambda i: K.expand_dims(i[:, 0], -1) + i[:, 1:] - K.mean(i[:, 1:], keepdims=True),
                        output_shape=(self.action_size,))(d41)
        else:
            d4 = Dense(self.action_size, activation='relu')(concat)

        model = Model(inputs=[input, input2], outputs=d4)
        model.summary()
        model.compile(Adam(self.learning_rate), loss=self._huber_loss, metrics=['accuracy'])
        return model

    # get random sample from experience pool
    def _get_samples(self):
        cache_len = len(self.cache)
        samples_count = cache_len if len(self.cache) < self.batch_size else self.batch_size
        # samples = random.sample(self.cache, samples_count)
        if self.weights is None:
            self.weights = np.ones(cache_len)
        elif len(self.weights) < cache_len:
            self.weights = np.concatenate([self.weights / 2, np.ones(cache_len - len(self.weights))])
        samples = sample_weighted(self.cache, self.weights, samples_count)
        # samples[0] == state == (observation, flags)
        state = (np.vstack([i[0][0] for i in samples]), np.vstack([i[0][1] for i in samples]))
        action = np.squeeze(np.vstack([i[1] for i in samples]))
        reward = np.squeeze(np.vstack([i[2] for i in samples]))
        next_state = (np.vstack([i[3][0] for i in samples]), np.vstack([i[3][1] for i in samples]))
        done = [i[4] for i in samples]
        return state, action, reward, next_state, done

    def get_deterministic_policy_batch(self, inputs):
        act_values = self.model_eval.predict(x={'state_eval': inputs[0], 'flag_eval': inputs[1]})
        return np.argmax(act_values, axis=1)

    def get_deterministic_policy(self, inputs):
        act_values = self.model_eval.predict(x={'state_eval': inputs[0], 'flag_eval': inputs[1]})
        return np.argmax(act_values[0])  # returns action

    def get_stochastic_policy(self, inputs):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model_eval.predict(x={'state_eval': inputs[0], 'flag_eval': inputs[1]})
        return np.argmax(act_values[0])  # returns action

    # update target network params
    def update_target_net(self):
        # copy weights from model to target_model
        self.model_target.set_weights(self.model_eval.get_weights())

    # update experience replay pool
    def update_cache(self, state, action, reward, next_state, done):
        self.cache.append((state, action, reward, next_state, done))

    # train, update value network params
    def update_value_net(self):
        state, action, reward, next_state, done = self._get_samples()
        inputs_eval = {'state_eval': next_state[0], 'flag_eval': next_state[1]}
        inputs_target = {'state_target': next_state[0], 'flag_target': next_state[1]}
        q_eval_next = self.model_eval.predict(x=inputs_eval)
        q_target_next = self.model_target.predict(x=inputs_target)
        q_target = self.model_target.predict(x={'state_target': state[0], 'flag_target': state[1]})
        done_arr = 1 - np.array(done).astype('int')
        index = np.arange(q_target_next.shape[0])
        q_target[index, action] = reward + done_arr * self.gamma * q_target_next[index, np.argmax(q_eval_next, axis=1)]

        if self.has_logged:
            self.model_eval.fit(inputs_eval, q_target_next, batch_size=self.batch_size, epochs=1,
                                verbose=0, callbacks=[self.fit_callback],
                                )
        else:
            self.model_eval.fit(inputs_eval, q_target_next, batch_size=self.batch_size, epochs=1,
                                verbose=0, callbacks=[TensorBoard(log_dir='./tensorboard_log'), self.fit_callback],
                                )
            self.has_logged = True
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return self.acc_loss_lists


def _test():
    from keras.utils import plot_model
    agent = Framework(input_shape=[None, 250, 78], action_size=4, dueling=True,
                      gamma=0.3, batch_size=512, memory_size=100000)
    file_path = 'model.png'
    plot_model(agent.model_eval, to_file=file_path, show_shapes=True)
    from ibats_utils.mess import open_file_with_system_app
    open_file_with_system_app(file_path)


if __name__ == '__main__':
    _test()
