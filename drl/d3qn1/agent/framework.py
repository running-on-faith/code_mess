import random
from collections import deque

import numpy as np
import tensorflow as tf

from ibats_common.example.drl.d3qn1.agent.forward import Forward

_EPSILON = 1e-6  # avoid nan


class Framework(object):
    def __init__(self, memory_size=2048, input_shape=[None, 50, 58, 5], max_grad_norm=10, action_size=3, batch_size=128,
                 gamma=0.9):

        self.memory_size = memory_size
        self.max_grad_norm = max_grad_norm
        self.action_size = action_size
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.gamma = gamma
        # placeholder
        self.inputs = tf.placeholder(tf.float32, input_shape, 'input')
        self.actions = tf.placeholder(tf.int32, [None], 'action')
        self.rewards = tf.placeholder(tf.float32, [None], 'reward')
        self.targets = tf.placeholder(tf.float32, [None], 'targets')

        # Q value eval
        self.value_eval = Forward('value', action_size=action_size)(self.inputs)

        # Q_ target eval
        value_next = tf.stop_gradient(self.value_eval)
        action_next = tf.one_hot(tf.argmax(value_next, axis=1), action_size)
        target_eval = Forward('target', action_size=action_size)(self.inputs)
        self.target_eval = tf.reduce_sum(target_eval * action_next, axis=1)

        # loss function
        action_choice = tf.one_hot(self.actions, action_size, 1., 0.)
        action_eval = tf.reduce_sum(
            self.value_eval * action_choice, axis=1)
        loss = tf.squared_difference(self.targets, action_eval)
        self._loss = tf.reduce_sum(loss)

        # train op
        trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'value')
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self._loss, trainable_variables), max_grad_norm)
        optimizer = tf.contrib.opt.NadamOptimizer()
        self._train_op = optimizer.apply_gradients(zip(grads, trainable_variables))

        # update target net params
        eval_net_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "value")
        target_net_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "target")
        self._target_params_swap = \
            [tf.assign(n, q) for n, q in zip(target_net_params, eval_net_params)]

        # cache for experience replay
        self.cache = deque(maxlen=memory_size)

    # get random sample from experience pool
    def _get_samples(self):
        samples_count = len(self.cache) if len(self.cache) < self.batch_size else self.batch_size
        samples = random.sample(self.cache, samples_count)
        state = np.vstack([i[0] for i in samples])
        action = np.squeeze(np.vstack([i[1] for i in samples]))
        reward = np.squeeze(np.vstack([i[2] for i in samples]))
        next_state = np.vstack([i[3] for i in samples])
        done = [i[4] for i in samples]
        return state, action, reward, next_state, done

    def get_deterministic_policy(self, sess, inputs):
        value_eval = sess.run(self.value_eval, {self.inputs: inputs})
        return np.argmax(value_eval, axis=1)[0]

    def get_stochastic_policy(self, sess, inputs, epsilon=0.9):
        if np.random.uniform() < epsilon:
            return self.get_deterministic_policy(sess, inputs)
        else:
            return np.random.randint(self.action_size)

    # update target network params
    def update_target_net(self, sess):
        sess.run(self._target_params_swap)

    # update experience replay pool
    def update_cache(self, state, action, reward, next_state, done):
        self.cache.append((state, action, reward, next_state, done))

    # train, update value network params
    def update_value_net(self, sess):
        state, action, reward, next_state, done = self._get_samples()
        mask = np.array(done).astype('float')

        target_eval = sess.run(self.target_eval, {self.inputs: next_state})
        target = mask * reward + (1 - mask) * (reward + self.gamma * target_eval)
        sess.run(self._train_op, {self.inputs: state, self.actions: action, self.targets: target})
