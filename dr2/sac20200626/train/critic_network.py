#! /usr/bin/env python3
"""
@author  : MG
@Time    : 2020/6/26 下午9:28
@File    : critic_network.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
import logging
import tensorflow as tf
from tensorflow.keras import activations, backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Lambda, Dropout, Concatenate
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks import network, utils
from tf_agents.networks.encoding_network import EncodingNetwork
from dr2.dqn20200209.train import plot_modal_2_file

logger = logging.getLogger()


class CriticLSTMNetwork(network.Network):

    def __init__(self, input_tensor_spec,
                 activation_fn=activations.tanh,
                 dueling=True,
                 name='CriticNetwork'):
        super().__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name)
        observation_spec, action_spec = input_tensor_spec
        value_count = 1  # Critic 网络输出的是状态与动作的价值评估,因此只有一个值
        # 设计 observation_spec 相关网络
        state_spec = observation_spec[0]
        input_shape = state_spec.shape[-1]
        unit = input_shape * 2
        _state_layer = LSTM(
            unit,
            dropout=0.2,
            recurrent_dropout=0.2)
        logger.debug("critic_network LSTM unit:%d", unit)
        _flag_layer = Lambda(lambda x: x)
        _rr_layer = Lambda(lambda x: x)
        preprocessing_layers = [_state_layer, _flag_layer, _rr_layer]

        self._encoder = EncodingNetwork(
            observation_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=Concatenate(axis=-1),
            activation_fn=activations.tanh,
        )

        # 设计 action_spec 相关网络
        flat_action_spec = tf.nest.flatten(action_spec)
        if len(flat_action_spec) > 1:
            raise ValueError('Only a single action is supported by this network')
        self._single_action_spec = flat_action_spec[0]

        seq_layer = Sequential()
        seq_layer.add(Dense(input_shape // 2, activation=activation_fn))
        seq_layer.add(Dropout(0.2))
        seq_layer.add(Dense(input_shape // 4, activation=activation_fn))
        seq_layer.add(Dropout(0.2))
        seq_layer.add(Dense(input_shape // 8, activation=activation_fn))
        seq_layer.add(Dropout(0.2))
        seq_layer.add(Dense(input_shape // 16, activation=activation_fn))
        seq_layer.add(Dropout(0.2))
        seq_layer.add(Dense(value_count, activation=activation_fn))

        self._joint_layer = seq_layer

    def call(self, inputs, step_type=(), network_state=(), training=False):
        observations, actions = inputs
        state, network_state = self._encoder(
            observations, step_type=step_type, network_state=network_state)
        # 合并 observations 与 actions
        joint = tf.concat([state, actions], 1)
        joint = self._joint_layer(joint, training=training)

        return tf.reshape(joint, [-1]), network_state


def get_critic_network(env: TFPyEnvironment, state_with_flag: bool, **kwargs):
    observation_spec, action_spec = env.observation_spec(), env.action_spec()
    net = CriticLSTMNetwork(
        input_tensor_spec=(observation_spec, action_spec), **kwargs
    )
    # plot_modal_2_file(net, 'critic.png')
    return net


if __name__ == "__main__":
    pass
