import tensorflow as tf
from pprint import pprint
import keraz.activations.classes as cls
from keraz.common import train_step

tf.random.set_seed(42)
true_weights = tf.constant(list(range(5)), dtype=tf.float32)[:, tf.newaxis]
x = tf.constant(tf.random.uniform((32, 5)), dtype=tf.float32)
y = tf.constant(x @ true_weights, dtype=tf.float32)

model = cls.NeuralNetworkV2([3, 1])
for iteration in range(5001):
    loss = train_step(x, y, model)
    if not (iteration % 1000):
        print('mean squared loss at iteration {:4d} is {:5.4f}'.format(iteration, loss))

print()
pprint(model.variables)
print('Mean absolute error is: ', tf.reduce_mean(tf.abs(y - model(x))).numpy())
