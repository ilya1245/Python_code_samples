import tensorflow as tf

@tf.function
def train_step(x, y, model):
    with tf.GradientTape() as tape:
        y_hat = model(x)
        loss = tf.reduce_mean(tf.square(y - y_hat))
    # gradient means derivative of a function (loss) for vars (model.variables)
    gradients = tape.gradient(loss, model.variables)
    for g, v in zip(gradients, model.variables):
        v.assign_add(tf.constant([-0.05], dtype=tf.float32) * g)
    return loss