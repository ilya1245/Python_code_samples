import time
import tensorflow as tf
from pprint import pprint
import keraz.layers.classes as cls

tf.random.set_seed(42)
true_weights = tf.constant(list(range(5)), dtype=tf.float32)[:, tf.newaxis]
x = tf.constant(tf.random.uniform((32, 5)), dtype=tf.float32)
y = tf.constant(x @ true_weights, dtype=tf.float32)


@tf.function
def train_step(model):
    with tf.GradientTape() as tape:
        y_hat = model(x)
        loss = tf.reduce_mean(tf.square(y - y_hat))
    # gradient means derivative of a function (loss) for vars (model.variables)
    gradients = tape.gradient(loss, model.variables)
    for g, v in zip(gradients, model.variables):
        v.assign_add(tf.constant([-0.05], dtype=tf.float32) * g)
    return loss


model = cls.RegressionV1([5, 1])
for iteration in range(5001):
    loss = train_step(model)
    if not (iteration % 1000):
        print('mean squared loss at iteration {:4d} is {:5.4f}'.format(iteration, loss))

print()
pprint(model.variables)
print('Mean absolute error is: ', tf.reduce_mean(tf.abs(y - model(x))).numpy())
