#! /usr/bin/env python3
"""
@author  : MG
@Time    : 2020/6/26 下午9:28
@File    : actor_network.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
import logging
from tensorflow.keras import activations
from tensorflow.keras.layers import LSTM, Lambda, Concatenate, Dense, Dropout
from tensorflow.keras import Sequential
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.sequential_layer import SequentialLayer

from dr2.dqn20200209.train import plot_modal_2_file

logger = logging.getLogger()


def get_actor_network(env: TFPyEnvironment, state_with_flag: bool, activation_fn=activations.tanh):
    observation_spec, action_spec = env.observation_spec(), env.action_spec()
    state_spec = observation_spec[0]
    input_shape = state_spec.shape[-1]
    unit = input_shape * 2
    _state_layer = SequentialLayer([LSTM(
        unit,
        dropout=0.2,
        recurrent_dropout=0.2, ),
        Dense(input_shape // 2, activation=activation_fn),
        Dropout(0.2),
        Dense(input_shape // 4, activation=activation_fn),
        Dropout(0.2),
        Dense(input_shape // 8, activation=activation_fn),
        Dropout(0.2),
        Dense(input_shape // 16, activation=activation_fn),
        Dropout(0.2),
        Dense(action_spec.shape.num_elements(), activation=activation_fn),
    ])
    logger.debug("actor_network LSTM unit:%d", unit)
    _flag_layer = Lambda(lambda x: x)
    _rr_layer = Lambda(lambda x: x)
    preprocessing_layers = [_state_layer, _flag_layer, _rr_layer]

    net = ActorDistributionNetwork(
        input_tensor_spec=observation_spec,
        output_tensor_spec=action_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=Concatenate(axis=-1),
    )
    # plot_modal_2_file(net, 'actor.png')
    return net


if __name__ == "__main__":
    pass
