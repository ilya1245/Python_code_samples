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


class LinearV4(tf.keras.layers.Layer):
    def __init__(self, units, use_bias=True, activation='linear', **kwargs):
        super(LinearV4, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.activation = activation

    def build(self, input_shape):
        self._weights = self.add_weight(shape=(input_shape[1], self.units))
        if self.use_bias:
            self._bias = self.add_weight(shape=(self.units), initializer='ones')
        super().build(input_shape)

    @tf.function
    def call(self, x):
        output = tf.linalg.matmul(x, self._weights)
        if self.use_bias:
            output += self._bias
        if self.activation == 'relu':
            output = tf.maximum(tf.constant(0, x.dtype), output)
        return output


class NeuralNetworkV2(tf.keras.Model):
    def __init__(self, units, use_bias=True, last_linear=True, **kwargs):
        super().__init__(**kwargs)
        layers = [LinearV4(unit, use_bias, 'relu') for unit in units[:-1]]
        layers.append(LinearV4(units[1], use_bias, 'linear' if last_linear else 'relu'))
        self._layers = layers

    @tf.function
    def call(self, x):
        for layer in self._layers:
            x = layer(x)
        return x
