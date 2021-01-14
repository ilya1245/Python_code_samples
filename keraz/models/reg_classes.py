import tensorflow as tf

print(tf.__version__)


class LinearRegressionV1(object):
    def __init__(self, num_parameters):
        self._weights = tf.Variable(tf.random.uniform((num_parameters, 1)), dtype=tf.float32)

    @tf.function
    def __call__(self, x):
        return tf.linalg.matmul(x, self._weights)

    @property
    def variables(self):
        return self._weights


class LinearRegressionV2(tf.keras.Model):
    def __init__(self, num_parameters, **kwargs):
        super().__init__(**kwargs)
        self._weights = tf.Variable(tf.random.uniform((num_parameters, 1)), dtype=tf.float32)

    @tf.function
    def call(self, x):
        return tf.linalg.matmul(x, self._weights)
