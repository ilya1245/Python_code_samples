import tensorflow as tf


class LinearV1(tf.keras.layers.Layer):
    def __init__(self, num_inputs, **kwargs):
        super().__init__(**kwargs)
        self._weights = tf.Variable(tf.random.uniform((num_inputs)), dtype=tf.float32)

    @tf.function
    def call(self, x):
        return tf.linalg.matmul(x, self._weights)


class RegressionV1(tf.keras.Model):
    def __init__(self, num_inputs_per_layer, **kwargs):
        super().__init__(**kwargs)
        self._layers = [LinearV1(num_inputs_per_layer)]

    @tf.function
    def call(self, x):
        for layer in self._layers:
            x = layer(x)
        return x


class LinearV2(tf.keras.layers.Layer):
    def __init__(self, num_inputs, num_outputs, **kwargs):
        super().__init__(**kwargs)
        self._weights = tf.Variable(tf.random.uniform((num_inputs, num_outputs)), dtype=tf.float32)

    @tf.function
    def call(self, x):
        return tf.linalg.matmul(x, self._weights)


class RegressionV2(tf.keras.Model):
    def __init__(self, num_inputs_per_layer, num_outputs_per_layer, **kwargs):
        super().__init__(**kwargs)

        # Two variants
        # self._layers = [LinearV2(_in, _out) for (_in, _out) in
        #                 zip(num_inputs_per_layer, num_outputs_per_layer)]
        self._layers = [LinearV2(_w[0], _w[1]) for _w in [num_inputs_per_layer, num_outputs_per_layer]]

    @tf.function
    def call(self, x):
        for layer in self._layers:
            x = layer(x)
        return x


# It is not clear, why RegressionV3 generates weights that different from RegressionV2
class LinearV3(tf.keras.layers.Layer):
    def __init__(self, unit, **kwargs):
        super(LinearV3, self).__init__(**kwargs)
        self.unit = unit

    def build(self, input_shape):
        self._weights = self.add_weight(shape=(input_shape[1], self.unit))
        # Works fine without super.build. Is is optional for subclasses
        # super().build(input_shape)

    @tf.function
    def call(self, x):
        output = tf.linalg.matmul(x, self._weights)
        return output


class RegressionV3(tf.keras.Model):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self._layers = [LinearV3(unit) for unit in units]

    @tf.function
    def call(self, x):
        for layer in self._layers:
            x = layer(x)
        return x


class RegressionD(tf.keras.Model):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self._layers = [tf.keras.layers.Dense(unit, use_bias=False) for unit in units]  # the only change

    @tf.function
    def call(self, x):
        for layer in self._layers:
            x = layer(x)
        return x
