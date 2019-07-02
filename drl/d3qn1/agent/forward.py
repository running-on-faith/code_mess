import tensorflow as tf
import sonnet as snt


def Swich(inputs):
    return inputs * tf.nn.sigmoid(inputs)


class Forward(snt.AbstractModule):
    def __init__(self, name='forward', action_size=3):
        super().__init__(name=name)
        self.name = name
        self.action_size = action_size

    def _build(self, inputs):
        with tf.variable_scope(self.name):
            net = self._build_shared_network(inputs)
            V = snt.Linear(1, 'value')(net)
            A = snt.Linear(self.action_size, 'advantage')(net)
            Q = V + (A - tf.reduce_mean(A, axis=1, keep_dims=True))
            return Q

    def _build_shared_network(self, inputs):
        net = snt.Conv2D(32, [8, 8], [4, 4])(inputs)
        net = Swich(net)
        net = snt.Conv2D(64, [4, 4], [2, 2])(net)
        net = Swich(net)
        net = snt.Conv2D(64, [3, 3], [1, 1])(net)
        net = Swich(net)
        net = snt.BatchFlatten(1)(net)
        net = snt.Linear(512)(net)
        return Swich(net)



