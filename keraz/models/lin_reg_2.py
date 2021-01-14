import time
import numpy as np
import pandas as pd
import tensorflow as tf
from functools import reduce
from matplotlib import pyplot as plt
from pprint import pprint
import keraz.models.reg_classes as cls

tf.random.set_seed(42)
true_weights = tf.constant(list(range(5)), dtype=tf.float32)[:, tf.newaxis]
x = tf.constant(tf.random.uniform((32, 5)), dtype=tf.float32)
y = tf.constant(x @ true_weights, dtype=tf.float32)

weights = tf.Variable(tf.random.uniform((5, 1)), dtype=tf.float32)
y_hat = tf.linalg.matmul(x, weights)

model = cls.LinearRegressionV2(5)
print('Use model_2')


@tf.function
def train_step():
    with tf.GradientTape() as tape:
        y_hat = model(x)
        loss = tf.reduce_mean(tf.square(y - y_hat))
    gradients = tape.gradient(loss, model.variables)
    for g, v in zip(gradients, model.variables):
        v.assign_add(tf.constant([-0.05], dtype=tf.float32) * g)
    return loss


t0 = time.time()
for iteration in range(1001):
    loss = train_step()
if not (iteration % 200):
    print('mean squared loss at iteration {:4d} is {:5.4f}'.format(iteration, loss))

pprint(model.variables)
print('time took: {} seconds'.format(time.time() - t0))

print(model.summary())
model.compile(loss='mse', metrics=['mae'])
print(model.evaluate(x, y, verbose=-1))
print('-----------------------------------')
print()

print('fit API usage')
model = cls.LinearRegressionV2(5)
model.compile(optimizer='SGD', loss='mse')
model.optimizer.lr.assign(.05)

t0 = time.time()
history = model.fit(x, y, epochs=1001, verbose=0)
pprint(history.history['loss'][::200])
pprint(model.variables)
print('time took: {} seconds'.format(time.time() - t0))
