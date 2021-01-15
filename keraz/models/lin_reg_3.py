import time
import tensorflow as tf
from pprint import pprint
import keraz.models.reg_classes as cls

tf.random.set_seed(42)
true_weights = tf.constant(list(range(5)), dtype=tf.float32)[:, tf.newaxis]
x = tf.constant(tf.random.uniform((32, 5)), dtype=tf.float32)
y = tf.constant(x @ true_weights, dtype=tf.float32)

weights = tf.Variable(tf.random.uniform((5, 1)), dtype=tf.float32)
y_hat = tf.linalg.matmul(x, weights)


@tf.function
def train_step(model):
    with tf.GradientTape() as tape:
        y_hat = model(x)
        loss = tf.reduce_mean(tf.square(y - y_hat))
    gradients = tape.gradient(loss, model.variables)
    for g, v in zip(gradients, model.variables):
        v.assign_add(tf.constant([-0.05], dtype=tf.float32) * g)
    return loss


print('Use model_3')
model = cls.LinearRegressionV3(5)

t0 = time.time()
for iteration in range(5001):
    loss = train_step(model)
pprint(model.variables)
print('time took: {} seconds'.format(time.time() - t0))

