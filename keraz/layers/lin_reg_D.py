import time
import tensorflow as tf
from pprint import pprint
import keraz.layers.classes as cls
from functools import reduce

tf.random.set_seed(42)
true_weights = tf.constant(list(range(5)), dtype=tf.float32)[:, tf.newaxis]
x = tf.constant(tf.random.uniform((32, 5)), dtype=tf.float32)
y = tf.constant(x @ true_weights, dtype=tf.float32)

# model = cls.RegressionD([1])  # brings weights 0, 1, 2, 3, 4
model = cls.RegressionD([3, 1])  # brings the same weights as RegressionV3
for iteration in range(1001):
    loss = cls.train_step(x, y, model)
    if not (iteration % 200):
        print('mean squared loss at iteration {:4d} is {:5.4f}'.format(iteration, loss))

print()
pprint(model.variables)
print('Mean absolute error is: ', tf.reduce_mean(tf.abs(y - model(x))).numpy())

print()
reduced_model = reduce(tf.linalg.matmul, model.variables)  # brings model with weights 0, 1, 2, 3, 4
print(reduced_model)
print(tf.reduce_all(tf.abs(model(x) - x @ reduced_model) < 1e-6))
