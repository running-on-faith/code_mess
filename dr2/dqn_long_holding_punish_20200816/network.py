#! /usr/bin/env python3
"""
@author  : MG
@Time    : 2020/8/16 上午9:38
@File    : network.py
@contact : mmmaaaggg@163.com
@desc    : 
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras import metrics, backend as backend
from tensorflow.keras.layers import Dense, LSTM, Lambda, Dropout, \
    Concatenate, BatchNormalization, RepeatVector, Reshape
from tf_agents.networks.encoding_network import CONV_TYPE_1D
from tf_agents.networks.network import Network
from tf_agents.networks.q_network import validate_specs
from tf_agents.networks import encoding_network
from tensorflow.keras.models import Sequential
from tensorflow.python.util import nest
from tf_agents.networks import utils


class DDQN(Network):

    def __init__(self,
                 input_tensor_spec,
                 action_spec,
                 lstm_kwargs,
                 conv_layer_params,
                 activation_fn=tf.keras.activations.sigmoid,
                 kernel_initializer=None,
                 batch_squash=True,
                 dtype=tf.float32,
                 name='QNetwork',
                 dueling=True,
                 learning_rate=0.001):
        """Creates an instance of `QNetwork`.

        Args:
          input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
            input observations.
          action_spec: A nest of `tensor_spec.BoundedTensorSpec` representing the
            actions.
          recurrent_dropout: a float number within range [0, 1). The ratio that the
            recurrent state weights need to dropout.
          fc_dropout_layer_params: Optional list of dropout layer parameters, where
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
        super().__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name)

        validate_specs(action_spec, input_tensor_spec)
        action_spec = tf.nest.flatten(action_spec)[0]
        action_count = action_spec.maximum - action_spec.minimum + 1
        self.dueling = dueling
        self.learning_rate = learning_rate
        state_spec = input_tensor_spec[0]
        input_shape = state_spec.shape[-1]
        self.bn = BatchNormalization(trainable=False)
        # Sequential 将会抛出异常:
        #   Weights for model sequential have not yet been created
        # layer = Sequential([
        #     LSTM(input_shape * 2, dropout=fc_dropout_layer_params, recurrent_dropout=recurrent_dropout),
        #     Dense(input_shape // 2, activation=activation_fn),
        #     Dropout(0.2),
        #     Dense(input_shape // 4, activation=activation_fn),
        #     Dropout(0.2),
        #     Dense(input_shape // 8, activation=activation_fn),
        #     Dropout(0.2),
        # ])
        self._state_layer = LSTM(input_shape * 2, **lstm_kwargs)
        self._flag_layer = RepeatVector(input_shape * 2)
        self._rr_layer = RepeatVector(input_shape * 2)
        preprocessing_layers = [self._state_layer, self._flag_layer, self._rr_layer]
        # Reshape((-1, input_shape * 2))(x[1]) -> output_shape = (0, 1, 186)
        # Concatenate(axis=1) -> output_shape = (0, 3, 186)
        preprocessing_combiner = Lambda(
            lambda x: Concatenate(axis=1)(
                [Reshape((-1, input_shape * 2))(x[1]),  # _flag_layer
                 Reshape((-1, input_shape * 2))(x[0]),  # _state_layer
                 Reshape((-1, input_shape * 2))(x[2]),  # _rr_layer
                 ]
            )
        )
        # (0, 3, 186) -> (0, 1, 372)
        # conv_layer_params = [
        #     ((input_shape * 2) * 2, 3, 2, 2),
        #     ((input_shape * 2) * 2, 3, 1),
        # ]
        # 形成一次递减的多个全连接层,比如,当前网络层数40,则向下将形成 [16, 8] 两层网络
        # fc_layers_counter_range = range(int(np.log2(input_shape * 2 * 2)) - 1, 2, -1)
        # fc_layer_params = [2 ** _ for _ in fc_layers_counter_range]
        # dropout_layer_params = [fc_dropout_layer_params for _ in fc_layers_counter_range]
        # fc_layer_params = [input_shape]
        # dropout_layer_params = [fc_dropout_layer_params]
        encoder = EnhanceEncodingNetwork(
            input_tensor_spec[:3],
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

        q_value_layer = Sequential()
        # 链接 encoder 的输出 和 observation[3] 持仓周期(holding_period)
        q_value_layer.add(Concatenate())
        # Dueling
        if dueling:
            q_value_layer.add(Dense(
                action_count + 1,
                activation=activation_fn
            ))
            q_value_layer.add(Lambda(
                lambda i: (backend.expand_dims(i[:, 0], -1) + i[:, 1:] -
                           backend.mean(i[:, 1:], keepdims=True)),
                output_shape=(action_count,)
            ))
        else:
            q_value_layer.add(Dense(
                action_count,
                activation=activation_fn,
                kernel_initializer='random_normal'
            ))

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
        # encoder_input, flag_input = observation
        new_obsercation = self.bn(observation[0]), observation[1], observation[2]
        state, network_state = self._encoder(
            new_obsercation, step_type=step_type, network_state=network_state)

        q_value = self._q_value_layer((state, observation[3]))
        try:
            q_numpy = q_value.numpy()
            if np.isnan(q_numpy).any():
                import logging
                logger = logging.getLogger()
                logger.warning("q_numpy=%s is nan. state=\n%s", q_numpy, state)
            elif q_numpy.shape == (1, 2) and q_numpy[0, 0] == q_numpy[0, 1]:
                import logging
                logger = logging.getLogger()
                logger.warning("q_numpy=%s is equal. state=\n%s", q_numpy, state)
        except AttributeError:
            import logging
            logger = logging.getLogger()
            logger.exception("q_value=%s. state=%s. It always happen at the beginning of drl training. Just ignore it",
                             q_value, state)
            pass

        return q_value, network_state


class EnhanceEncodingNetwork(encoding_network.EncodingNetwork):
    def __init__(self,
                 input_tensor_spec,
                 preprocessing_layers=None,
                 preprocessing_combiner=None,
                 conv_layer_params=None,
                 fc_layer_params=None,
                 dropout_layer_params=None,
                 activation_fn=tf.keras.activations.tanh,
                 weight_decay_params=None,
                 kernel_initializer=None,
                 batch_squash=True,
                 dtype=tf.float32,
                 name='EncodingNetwork',
                 conv_type=encoding_network.CONV_TYPE_1D):
        """Creates an instance of `EncodingNetwork`.

        Network supports calls with shape outer_rank + input_tensor_spec.shape. Note
        outer_rank must be at least 1.

        For example an input tensor spec with shape `(2, 3)` will require
        inputs with at least a batch size, the input shape is `(?, 2, 3)`.

        Input preprocessing is possible via `preprocessing_layers` and
        `preprocessing_combiner` Layers.  If the `preprocessing_layers` nest is
        shallower than `input_tensor_spec`, then the layers will get the subnests.
        For example, if:

        ```python
        input_tensor_spec = ([TensorSpec(3)] * 2, [TensorSpec(3)] * 5)
        preprocessing_layers = (Layer1(), Layer2())
        ```

        then preprocessing will call:

        ```python
        preprocessed = [preprocessing_layers[0](observations[0]),
                        preprocessing_layers[1](obsrevations[1])]
        ```

        However if

        ```python
        preprocessing_layers = ([Layer1() for _ in range(2)],
                                [Layer2() for _ in range(5)])
        ```

        then preprocessing will call:
        ```python
        preprocessed = [
          layer(obs) for layer, obs in zip(flatten(preprocessing_layers),
                                           flatten(observations))
        ]
        ```

        **NOTE** `preprocessing_layers` and `preprocessing_combiner` are not allowed
        to have already been built.  This ensures calls to `network.copy()` in the
        future always have an unbuilt, fresh set of parameters.  Furtheremore,
        a shallow copy of the layers is always created by the Network, so the
        layer objects passed to the network are never modified.  For more details
        of the semantics of `copy`, see the docstring of
        `tf_agents.networks.Network.copy`.

        Args:
          input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
            input observations.
          preprocessing_layers: (Optional.) A nest of `tf.keras.layers.Layer`
            representing preprocessing for the different observations. All of these
            layers must not be already built.
          preprocessing_combiner: (Optional.) A keras layer that takes a flat list
            of tensors and combines them.  Good options include
            `tf.keras.layers.Add` and `tf.keras.layers.Concatenate(axis=-1)`. This
            layer must not be already built.
          conv_layer_params: Optional list of convolution layers parameters, where
            each item is either a length-three tuple indicating
            `(filters, kernel_size, stride)` or a length-four tuple indicating
            `(filters, kernel_size, stride, dilation_rate)`.
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
          activation_fn: Activation function, e.g. tf.keras.activations.relu.
          weight_decay_params: Optional list of weight decay parameters for the
            fully connected layers.
          kernel_initializer: Initializer to use for the kernels of the conv and
            dense layers. If none is provided a default variance_scaling_initializer
          batch_squash: If True the outer_ranks of the observation are squashed into
            the batch dimension. This allow encoding networks to be used with
            observations with shape [BxTx...].
          dtype: The dtype to use by the convolution and fully connected layers.
          name: A string representing name of the network.
          conv_type: string, '1d' or '2d'. Convolution layers will be 1d or 2D
            respectively

        Raises:
          ValueError: If any of `preprocessing_layers` is already built.
          ValueError: If `preprocessing_combiner` is already built.
          ValueError: If the number of dropout layer parameters does not match the
            number of fully connected layer parameters.
          ValueError: If conv_layer_params tuples do not have 3 or 4 elements each.
        """
        if preprocessing_layers is None:
            flat_preprocessing_layers = None
        else:
            flat_preprocessing_layers = [
                encoding_network._copy_layer(layer) for layer in tf.nest.flatten(preprocessing_layers)
            ]
            # Assert shallow structure is the same. This verifies preprocessing
            # layers can be applied on expected input nests.
            input_nest = input_tensor_spec
            # Given the flatten on preprocessing_layers above we need to make sure
            # input_tensor_spec is a sequence for the shallow_structure check below
            # to work.
            if not nest.is_sequence(input_tensor_spec):
                input_nest = [input_tensor_spec]
            nest.assert_shallow_structure(
                preprocessing_layers, input_nest, check_types=False)

        if (len(tf.nest.flatten(input_tensor_spec)) > 1 and
                preprocessing_combiner is None):
            raise ValueError(
                'preprocessing_combiner layer is required when more than 1 '
                'input_tensor_spec is provided.')

        if preprocessing_combiner is not None:
            preprocessing_combiner = encoding_network._copy_layer(preprocessing_combiner)

        if not kernel_initializer:
            kernel_initializer = tf.compat.v1.variance_scaling_initializer(
                scale=2.0, mode='fan_in', distribution='truncated_normal')

        layers = []

        if conv_layer_params:
            if conv_type == '2d':
                conv_layer_type = tf.keras.layers.Conv2D
            elif conv_type == '1d':
                conv_layer_type = tf.keras.layers.Conv1D
            else:
                raise ValueError('unsupported conv type of %s. Use 1d or 2d' % (
                    conv_type))

            for config in conv_layer_params:
                if len(config) == 5:
                    (filters, kernel_size, strides, dilation_rate, padding) = config
                elif len(config) == 4:
                    (filters, kernel_size, strides, dilation_rate) = config
                    padding = 'valid'
                elif len(config) == 3:
                    (filters, kernel_size, strides) = config
                    dilation_rate = (1, 1) if conv_type == '2d' else (1,)
                    padding = 'valid'
                else:
                    raise ValueError(
                        'only 3 or 4 elements permitted in conv_layer_params tuples')
                layers.append(
                    conv_layer_type(
                        filters=filters,
                        kernel_size=kernel_size,
                        strides=strides,
                        dilation_rate=dilation_rate,
                        padding=padding,
                        activation=activation_fn,
                        kernel_initializer=kernel_initializer,
                        dtype=dtype))

        layers.append(tf.keras.layers.Flatten())

        if fc_layer_params:
            if dropout_layer_params is None:
                dropout_layer_params = [None] * len(fc_layer_params)
            else:
                if len(dropout_layer_params) != len(fc_layer_params):
                    raise ValueError('Dropout and fully connected layer parameter lists'
                                     'have different lengths (%d vs. %d.)' %
                                     (len(dropout_layer_params), len(fc_layer_params)))
            if weight_decay_params is None:
                weight_decay_params = [None] * len(fc_layer_params)
            else:
                if len(weight_decay_params) != len(fc_layer_params):
                    raise ValueError('Weight decay and fully connected layer parameter '
                                     'lists have different lengths (%d vs. %d.)' %
                                     (len(weight_decay_params), len(fc_layer_params)))

            for num_units, dropout_params, weight_decay in zip(
                    fc_layer_params, dropout_layer_params, weight_decay_params):
                kernal_regularizer = None
                if weight_decay is not None:
                    kernal_regularizer = tf.keras.regularizers.l2(weight_decay)
                layers.append(
                    tf.keras.layers.Dense(
                        num_units,
                        activation=activation_fn,
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernal_regularizer,
                        dtype=dtype))
                if not isinstance(dropout_params, dict):
                    dropout_params = {'rate': dropout_params} if dropout_params else None

                if dropout_params is not None:
                    layers.append(utils.maybe_permanent_dropout(**dropout_params))

        super().__init__(
            input_tensor_spec=input_tensor_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            weight_decay_params=weight_decay_params,
            kernel_initializer=kernel_initializer,
            batch_squash=batch_squash,
            dtype=dtype,
            name=name,
            conv_type=conv_type
        )

        # Pull out the nest structure of the preprocessing layers. This avoids
        # saving the original kwarg layers as a class attribute which Keras would
        # then track.
        self._preprocessing_nest = tf.nest.map_structure(lambda l: None,
                                                         preprocessing_layers)
        self._flat_preprocessing_layers = flat_preprocessing_layers
        self._preprocessing_combiner = preprocessing_combiner
        self._postprocessing_layers = layers
        self._batch_squash = batch_squash


def get_network(observation_spec, action_spec, network_kwargs_func=None, **kwargs):
    from tf_agents.utils import common
    # TODO: fc_layer_params 需要参数化
    if network_kwargs_func is not None:
        network_kwargs = network_kwargs_func(observation_spec, action_spec)
        kwargs.update(network_kwargs)

    network = DDQN(observation_spec, action_spec, **kwargs)
    learning_rate = 1e-3
    # optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer = Nadam(learning_rate)

    def _huber_loss(y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = backend.abs(error) <= clip_delta

        squared_loss = 0.5 * backend.square(error)
        quadratic_loss = 0.5 * backend.square(clip_delta) + (
                clip_delta * (backend.abs(error) - clip_delta))

        return backend.mean(tf.where(cond, squared_loss, quadratic_loss))

    # loss_fn = _huber_loss
    loss_fn = common.element_wise_squared_loss
    # loss_fn = common.element_wise_huber_loss
    return network, optimizer, loss_fn, kwargs


if __name__ == "__main__":
    pass
