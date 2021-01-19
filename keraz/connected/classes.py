import tensorflow as tf
from keraz.layers.classes import LinearV3


class Sequential(tf.keras.Model):
    def __init__(self, layers, **kwargs):
        super().__init__(**kwargs)
        self._layers = layers

    @tf.function
    def call(self, x):
        for layer in self._layers:
            x = layer(x)
        return x


class MLP(tf.keras.Model):
    def __init__(self, num_hidden_units, num_targets, hidden_activation='relu', **kwargs):
        super().__init__(**kwargs)
        if type(num_hidden_units) is int: num_hidden_units = [num_hidden_units]
        self.feature_extractor = Sequential([tf.keras.layers.Dense(unit, activation=hidden_activation)
                                             for unit in num_hidden_units])
        self.last_linear = tf.keras.layers.Dense(num_targets, activation='linear')

    @tf.function
    def call(self, x):
        features = self.feature_extractor(x)
        outputs = self.last_linear(features)
        return outputs