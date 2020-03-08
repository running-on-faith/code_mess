#! /usr/bin/env python3
"""
@author  : MG
@Time    : 2020/2/12 上午9:46
@File    : network.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
import tensorflow as tf
from tf_agents.networks.network import Network
from tf_agents.networks.q_network import validate_specs
from tf_agents.networks import encoding_network


class DQN(Network):

    def __init__(self,
                 input_tensor_spec,
                 action_spec,
                 conv_layer_params=None,
                 fc_layer_params=(75, 40),
                 dropout_layer_params=None,
                 activation_fn=tf.keras.activations.relu,
                 kernel_initializer=None,
                 batch_squash=True,
                 dtype=tf.float32,
                 name='QNetwork',
                 state_with_flag=False):
        """Creates an instance of `QNetwork`.

        Args:
          input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
            input observations.
          action_spec: A nest of `tensor_spec.BoundedTensorSpec` representing the
            actions.
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
          dropout_layer_params: Optional list of dropout layer parameters, where
            each item is the fraction of input units to drop. The dropout layers are
            interleaved with the fully connected layers; there is a dropout layer
            after each fully connected layer, except if the entry in the list is
            None. This list must have the same length of fc_layer_params, or be
            None.
          activation_fn: Activation function, e.g. tf.keras.activations.relu.
          kernel_initializer: Initializer to use for the kernels of the conv and
            dense layers. If none is provided a default variance_scaling_initializer
          batch_squash: If True the outer_ranks of the observation are squashed into
            the batch dimension. This allow encoding networks to be used with
            observations with shape [BxTx...].
          dtype: The dtype to use by the convolution and fully connected layers.
          name: A string representing the name of the network.

        Raises:
          ValueError: If `input_tensor_spec` contains more than one observation. Or
            if `action_spec` contains more than one action.
        """
        validate_specs(action_spec, input_tensor_spec)
        action_spec = tf.nest.flatten(action_spec)[0]
        num_actions = action_spec.maximum - action_spec.minimum + 1
        # if state_with_flag=True.
        # input_tensor_spec =
        # [
        #   TensorSpec(shape=(60, 79), dtype=tf.float64, name='state'),
        #   BoundedTensorSpec(shape=(1,), dtype=tf.float32, name='flag',
        #     minimum=array(0., dtype=float32), maximum=array(1., dtype=float32))
        # ]
        self.state_with_flag = state_with_flag
        if state_with_flag:
            preprocessing_layers = None,
            preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)
        else:
            preprocessing_layers = None,
            preprocessing_combiner = None,

        encoder = encoding_network.EncodingNetwork(
            input_tensor_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            dropout_layer_params=dropout_layer_params,
            activation_fn=activation_fn,
            kernel_initializer=kernel_initializer,
            batch_squash=batch_squash,
            dtype=dtype)

        q_value_layer = tf.keras.layers.Dense(
            num_actions,
            activation=None,
            kernel_initializer=tf.compat.v1.initializers.random_uniform(
                minval=-0.03, maxval=0.03),
            bias_initializer=tf.compat.v1.initializers.constant(-0.2),
            dtype=dtype)

        super().__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name)

        self._encoder = encoder
        self._q_value_layer = q_value_layer

    def call(self, observation, step_type=None, network_state=()):
        """Runs the given observation through the network.

        Args:
          observation: The observation to provide to the network.
          step_type: The step type for the given observation. See `StepType` in
            time_step.py.
          network_state: A state tuple to pass to the network, mainly used by RNNs.

        Returns:
          A tuple `(logits, network_state)`.
        """
        if self.state_with_flag:
            encoder_input, flag_input = observation
            state, network_state = self._encoder(
                encoder_input, step_type=step_type, network_state=network_state)
        else:
            state, network_state = self._encoder(
            observation, step_type=step_type, network_state=network_state)

        return self._q_value_layer(state), network_state


def get_network(observation_spec, action_spec, **kwargs):
    from tf_agents.networks.q_network import QNetwork
    from tf_agents.utils import common
    network = DQN(observation_spec, action_spec, fc_layer_params=(100,), **kwargs)
    learning_rate = 1e-3
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    loss_fn = common.element_wise_squared_loss
    return network, optimizer, loss_fn


if __name__ == "__main__":
    pass
