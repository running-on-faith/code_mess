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
from dr2.dqn20200209.train.network import EnhanceEncodingNetwork
import gin
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.networks import categorical_projection_network
from tf_agents.networks import encoding_network
from tf_agents.networks import network
from tf_agents.networks import normal_projection_network
from tf_agents.specs import tensor_spec
from tf_agents.utils import nest_utils
from tf_agents.networks.actor_distribution_network import _categorical_projection_net, _normal_projection_net

logger = logging.getLogger()


@gin.configurable
class ActorDistributionLSTMNetwork(network.DistributionNetwork):
    """Creates an actor producing either Normal or Categorical distribution.

    Note: By default, this network uses `NormalProjectionNetwork` for continuous
    projection which by default uses `tanh_squash_to_spec` to normalize its
    output. Due to the nature of the `tanh` function, values near the spec bounds
    cannot be returned.
    """

    def __init__(self,
                 input_tensor_spec,
                 output_tensor_spec,
                 lstm_kwargs,
                 preprocessing_layers=None,
                 preprocessing_combiner=None,
                 conv_layer_params=None,
                 fc_layer_params=(200, 100),
                 dropout_layer_params=None,
                 activation_fn=tf.keras.activations.relu,
                 kernel_initializer=None,
                 batch_squash=True,
                 dtype=tf.float32,
                 discrete_projection_net=_categorical_projection_net,
                 continuous_projection_net=_normal_projection_net,
                 name='ActorDistributionNetwork',
                 **kwargs):
        """Creates an instance of `ActorDistributionNetwork`.

        Args:
          input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
            input.
          output_tensor_spec: A nest of `tensor_spec.BoundedTensorSpec` representing
            the output.
          preprocessing_layers: (Optional.) A nest of `tf.keras.layers.Layer`
            representing preprocessing for the different observations.
            All of these layers must not be already built. For more details see
            the documentation of `networks.EncodingNetwork`.
          preprocessing_combiner: (Optional.) A keras layer that takes a flat list
            of tensors and combines them. Good options include
            `tf.keras.layers.Add` and `tf.keras.layers.Concatenate(axis=-1)`.
            This layer must not be already built. For more details see
            the documentation of `networks.EncodingNetwork`.
          conv_layer_params: Optional list of convolution layers parameters, where
            each item is a length-three tuple indicating (filters, kernel_size,
            stride).
          fc_layer_params: Optional list of fully_connected parameters, where each
            item is the number of units in the layer.
          dropout_layer_params: Optional list of dropout layer parameters, each item
            is the fraction of input units to drop or a dictionary of parameters
            according to the keras.Dropout documentation. The additional parameter
            `permanent', if set to True, allows to apply dropout at inference for
            approximated Bayesian inference. The dropout layers are interleaved with
            the fully connected layers; there is a dropout layer after each fully
            connected layer, except if the entry in the list is None. This list must
            have the same length of fc_layer_params, or be None.
          activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu, ...
          kernel_initializer: Initializer to use for the kernels of the conv and
            dense layers. If none is provided a default glorot_uniform
          batch_squash: If True the outer_ranks of the observation are squashed into
            the batch dimension. This allow encoding networks to be used with
            observations with shape [BxTx...].
          dtype: The dtype to use by the convolution and fully connected layers.
          discrete_projection_net: Callable that generates a discrete projection
            network to be called with some hidden state and the outer_rank of the
            state.
          continuous_projection_net: Callable that generates a continuous projection
            network to be called with some hidden state and the outer_rank of the
            state.
          name: A string representing name of the network.

        Raises:
          ValueError: If `input_tensor_spec` contains more than one observation.
        """

        def map_proj(spec):
            if tensor_spec.is_discrete(spec):
                return discrete_projection_net(spec)
            else:
                return continuous_projection_net(spec)

        projection_networks = tf.nest.map_structure(map_proj, output_tensor_spec)
        output_spec = tf.nest.map_structure(lambda proj_net: proj_net.output_spec,
                                            projection_networks)

        super().__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            output_spec=output_spec,
            name=name)

        if not kernel_initializer:
            kernel_initializer = tf.compat.v1.keras.initializers.glorot_uniform()

        # 设计 observation_spec 相关网络
        state_spec = input_tensor_spec[0]
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
        encoder = EnhanceEncodingNetwork(
            input_tensor_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            name=name,
            **kwargs
        )
        self._encoder = encoder
        self._projection_networks = projection_networks
        self._output_tensor_spec = output_tensor_spec

    @property
    def output_tensor_spec(self):
        return self._output_tensor_spec

    def call(self,
             observations,
             step_type,
             network_state,
             training=False,
             mask=None):
        state, network_state = self._encoder(
            observations,
            step_type=step_type,
            network_state=network_state,
            training=training)
        outer_rank = nest_utils.get_outer_rank(observations, self.input_tensor_spec)

        def call_projection_net(proj_net):
            distribution, _ = proj_net(
                state, outer_rank, training=training, mask=mask)
            return distribution

        output_actions = tf.nest.map_structure(
            call_projection_net, self._projection_networks)
        return output_actions, network_state


def get_actor_network(env: TFPyEnvironment, state_with_flag: bool, **kwargs):
    observation_spec, action_spec = env.observation_spec(), env.action_spec()
    net = ActorDistributionLSTMNetwork(
        input_tensor_spec=observation_spec,
        output_tensor_spec=action_spec,
        **kwargs
    )
    # plot_modal_2_file(net, 'actor.png')
    return net


if __name__ == "__main__":
    pass
