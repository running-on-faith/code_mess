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
from tensorflow.keras.layers import Dense, LSTM, Lambda, Dropout, \
    Concatenate, BatchNormalization, RepeatVector, Reshape
from tf_agents.networks.encoding_network import CONV_TYPE_1D
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks import network, utils
from tf_agents.networks.encoding_network import EncodingNetwork
from dr2.dqn20200209.train import plot_modal_2_file
from dr2.dqn20200209.train.network import EnhanceEncodingNetwork

logger = logging.getLogger()


class CriticLSTMNetwork(network.Network):

    def __init__(self,
                 input_tensor_spec,
                 lstm_kwargs,
                 conv_layer_params,
                 activation_fn=tf.keras.activations.sigmoid,
                 kernel_initializer=None,
                 batch_squash=True,
                 dtype=tf.float32,
                 name='CriticNetwork'):
        super().__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name)
        observation_spec, action_spec = input_tensor_spec
        value_count = 1  # Critic 网络输出的时状态与动作的价值评估,因此只有一个值
        # 设计 observation_spec 相关网络
        state_spec = observation_spec[0]
        input_shape = state_spec.shape[-1]
        unit = input_shape * 2
        lstm_kwargs = {} if lstm_kwargs is None else lstm_kwargs
        _state_layer = LSTM(unit, **lstm_kwargs)
        _flag_layer = RepeatVector(input_shape * 2)
        _rr_layer = RepeatVector(input_shape * 2)
        preprocessing_layers = [_state_layer, _flag_layer, _rr_layer]
        preprocessing_combiner = Lambda(
            lambda x: Concatenate(axis=1)(
                [Reshape((-1, input_shape * 2))(x[1]),
                 Reshape((-1, input_shape * 2))(x[0]),
                 Reshape((-1, input_shape * 2))(x[2])]
            )
        )
        self._encoder = EnhanceEncodingNetwork(
            observation_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            conv_type=CONV_TYPE_1D,
            # fc_layer_params=fc_layer_params,
            # dropout_layer_params=dropout_layer_params,
            activation_fn=activation_fn,
            kernel_initializer=kernel_initializer,
            batch_squash=batch_squash,
            dtype=dtype
        )

        # 设计 action_spec 相关网络
        flat_action_spec = tf.nest.flatten(action_spec)
        if len(flat_action_spec) > 1:
            raise ValueError('Only a single action is supported by this network')
        self._single_action_spec = flat_action_spec[0]

        self._joint_layer = Dense(value_count, activation=activation_fn)

    def call(self, inputs, step_type=(), network_state=(), training=False):
        observations, actions = inputs
        state, network_state = self._encoder(
            observations, step_type=step_type, network_state=network_state)
        # 合并 observations 与 actions
        joint = tf.concat([state, actions], 1)
        joint = self._joint_layer(joint, training=training)

        return tf.reshape(joint, [-1]), network_state


def get_critic_network(env: TFPyEnvironment, state_with_flag: bool, critic_net_kwargs_func=None, **kwargs):
    observation_spec, action_spec = env.observation_spec(), env.action_spec()
    kwargs.update(**critic_net_kwargs_func(observation_spec, action_spec))
    net = CriticLSTMNetwork(
        input_tensor_spec=(observation_spec, action_spec), **kwargs
    )
    # plot_modal_2_file(net, 'critic.png')
    return net


if __name__ == "__main__":
    pass
