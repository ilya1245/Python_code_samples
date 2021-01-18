import tensorflow as tf
from keraz.layers.classes import LinearV3

class ReLU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    @tf.function
    def call(self, x):
        return tf.maximum(tf.constant(0, x.dtype), x)


class NeuralNetworkV1(tf.keras.Model):
    def __init__(self, units, last_linear=True, **kwargs):
        super().__init__(**kwargs)
        layers = []
        n = len(units)
        for i, unit in enumerate(units):
            layers.append(LinearV3(unit))
            # Add ReLU layer after each layer except the last one.
            if i < n - 1 or not last_linear:
                layers.append(ReLU())
        self._layers = layers

    @tf.function
    def call(self, x):
        for layer in self._layers:
            x = layer(x)
        return x