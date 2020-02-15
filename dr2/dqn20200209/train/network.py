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


class DQN(Network):

    def __init__(self, input_tensor_spec, name):
        """
        Creates an instance of `Network`.

        Args:
          input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
            input observations.
          name: A string representing the name of the network.
        """
        super().__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name)

    def call(self, observation, step_type=None, network_state=(), training=False):
        """Runs the given observation through the network.

        Args:
          observation: The observation to provide to the network.
          step_type: The step type for the given observation. See `StepType` in
            time_step.py.
          network_state: A state tuple to pass to the network, mainly used by RNNs.
          training: Whether the output is being used for training.

        Returns:
          A tuple `(logits, network_state)`.
        """
        pass


def get_network(observation_spec, action_spec, **kwargs):
    from tf_agents.networks.q_network import QNetwork
    from tf_agents.utils import common
    network = QNetwork(observation_spec, action_spec, fc_layer_params=(100,), **kwargs)
    learning_rate = 1e-3
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    loss_fn = common.element_wise_squared_loss
    return network, optimizer, loss_fn


if __name__ == "__main__":
    pass
