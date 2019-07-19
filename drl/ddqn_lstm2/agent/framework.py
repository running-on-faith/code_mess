import logging
import random
from collections import deque

import numpy as np
import tensorflow as tf
from ibats_utils.mess import sample_weighted
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.layers import Dense, LSTM, Dropout, Input, concatenate
from keras.models import Model
from keras.optimizers import Adam

_EPSILON = 1e-6  # avoid nan


class Framework(object):
    def __init__(self, memory_size=2048, input_shape=[None, 50, 58, 5], max_grad_norm=10, action_size=4, batch_size=512,
                 gamma=0.95, learning_rate=0.001, tensorboard_log_dir='./log'):

        self.memory_size = memory_size
        self.input_shape = input_shape
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.tensorboard_log_dir = tensorboard_log_dir

        # cache for experience replay
        self.cache = deque(maxlen=memory_size)
        self.weights = None  # 用于 _get_samples 基于权重提取数据
        self.logger = logging.getLogger(str(self.__class__))
        K.clear_session()
        tf.reset_default_graph()

        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.3
        self.epsilon_decay = 0.9998
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_net()
        self.has_logged = False

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        input = Input(batch_shape=self.input_shape, name=f'state')
        lstm = LSTM(self.input_shape[-1] * 2, return_sequences=True, activation='relu')(input)
        lstm = LSTM(self.input_shape[-1], return_sequences=False, activation='relu')(lstm)
        d1 = Dense(self.input_shape[-1] * 2)(lstm)
        dr1 = Dropout(0.4)(d1)
        d2 = Dense(self.input_shape[-1])(dr1)
        dr2 = Dropout(0.4)(d2)
        d3 = Dense(self.action_size * 2, activation='relu')(dr2)
        input2 = Input(batch_shape=[None, 1], name=f'flag')
        concat = concatenate([d3, input2])
        d4 = Dense(self.action_size, activation='relu')(concat)

        model = Model(inputs=[input, input2], outputs=d4)
        model.summary()
        model.compile(Adam(self.learning_rate), loss=self._huber_loss)
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
        act_values = self.model.predict(x={'state': inputs[0], 'flag': inputs[1]})
        return np.argmax(act_values, axis=1)

    def get_deterministic_policy(self, inputs):
        act_values = self.model.predict(x={'state': inputs[0], 'flag': inputs[1]})
        return np.argmax(act_values[0])  # returns action

    def get_stochastic_policy(self, inputs):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(x={'state': inputs[0], 'flag': inputs[1]})
        return np.argmax(act_values[0])  # returns action

    # update target network params
    def update_target_net(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    # update experience replay pool
    def update_cache(self, state, action, reward, next_state, done):
        self.cache.append((state, action, reward, next_state, done))

    # train, update value network params
    def update_value_net(self):
        state, action, reward, next_state, done = self._get_samples()
        inputs = {'state': state[0], 'flag': state[1]}
        target = self.model.predict(x=inputs)
        t = self.target_model.predict(x=inputs)
        done_arr = 1 - np.array(done).astype('int')
        target[np.arange(target.shape[0]), action] = reward + done_arr * self.gamma * np.amax(t, axis=1)

        # if done:
        #     target[0][action] = reward
        # else:
        #     # a = self.model.predict([next_state,matrix])[0]
        #     t = self.target_model.predict([next_state, matrix])[0]
        #     target[0][action] = reward + self.gamma * np.amax(t)
        #     # target[0][action] = reward + self.gamma * t[np.argmax(a)]
        # , verbose=0, callbacks=[TensorBoard(log_dir='./tmp/log')]
        if self.has_logged:
            self.model.fit(inputs, target, batch_size=self.batch_size, epochs=1,
                           verbose=0,
                           )
        else:
            self.model.fit(inputs, target, batch_size=self.batch_size, epochs=1,
                           verbose=0,
                           callbacks=[TensorBoard(log_dir='./tmp/log')],
                           )
            self.has_logged = True
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
