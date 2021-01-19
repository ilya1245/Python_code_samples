import time
import numpy as np
import pandas as pd
import tensorflow as tf
from functools import reduce
from matplotlib import pyplot as plt
from pprint import pprint
from keraz.common import train_step
from keraz.connected.classes import MLP


(x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.boston_housing.load_data()
y_tr, y_te = map(lambda x: np.expand_dims(x, -1), (y_tr, y_te))
x_tr, y_tr, x_te, y_te = map(lambda x: tf.cast(x, tf.float32), (x_tr, y_tr, x_te, y_te))


@tf.function
def test_step(x, y, model):
    return tf.reduce_mean(tf.square(y - model(x)))

def train(model, n_epochs=1000, his_freq=10):
    history = []
    for iteration in range(1, n_epochs + 1):
        tr_loss = train_step(x_tr, y_tr, model, -0.01)
        te_loss = test_step(x_te, y_te, model)
        if not iteration % his_freq:
            history.append({
                'iteration': iteration,
                'training_loss': tr_loss.numpy(),
                'testing_loss': te_loss.numpy()
            })
    return model, pd.DataFrame(history)

mlp, mlp_history = train(MLP(4, 1))
pprint(mlp_history.tail())
ax = mlp_history.plot(x='iteration', kind='line', logy=True)
fig = ax.get_figure()
fig.savefig('ch3_plot_1.png')