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
from tensorflow.keras.layers import Dense, LSTM, Lambda, Dropout, \
    Concatenate, BatchNormalization, RepeatVector, Reshape
from tensorflow.keras import Sequential
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.sequential_layer import SequentialLayer

from dr2.dqn20200209.train import plot_modal_2_file

logger = logging.getLogger()


def get_actor_network(env: TFPyEnvironment, state_with_flag: bool, lstm_kwargs, actor_net_kwargs_func, **kwargs):
    observation_spec, action_spec = env.observation_spec(), env.action_spec()
    state_spec = observation_spec[0]
    input_shape = state_spec.shape[-1]
    unit = input_shape * 2
    lstm_kwargs = {} if lstm_kwargs is None else lstm_kwargs
    _state_layer = LSTM(unit, **lstm_kwargs)
    _flag_layer = RepeatVector(unit)
    _rr_layer = RepeatVector(unit)
    preprocessing_layers = [_state_layer, _flag_layer, _rr_layer]
    preprocessing_combiner = Lambda(
        lambda x: Concatenate(axis=1)(
            [Reshape((-1, input_shape * 2))(x[1]),
             Reshape((-1, input_shape * 2))(x[0]),
             Reshape((-1, input_shape * 2))(x[2])]
        )
    )
    net = ActorDistributionNetwork(
        input_tensor_spec=observation_spec,
        output_tensor_spec=action_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        **actor_net_kwargs_func(observation_spec, action_spec)
    )
    # plot_modal_2_file(net, 'actor.png')
    return net


if __name__ == "__main__":
    pass
