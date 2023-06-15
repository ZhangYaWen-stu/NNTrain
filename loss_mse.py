import tensorflow as tf
import numpy as np


def loss_mse():
    x = np.random.rand(32, 2)
    y = [[x1 + x2 + np.random.rand() / 10.0 - 0.05] for (x1, x2) in x]
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.float32)
    w = tf.Variable(tf.random.truncated_normal([2, 1], stddev=2))
    epochs = 20000
    lr = 0.001
    COST = 15
    PROFIT = 20
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            y_ = tf.matmul(x, w)
            # loss=tf.reduce_mean(tf.square(y-y_))
            loss = tf.reduce_sum(tf.where(tf.greater(y_, y), COST * (y_ - y), PROFIT * (y - y_)))
            grad = tape.gradient(loss, w)
            w.assign_sub(lr * grad)
        if epoch % 500 == 0:
            print("epoch:%d  loss:%f   w:" % (epoch, loss))
            print(w.numpy())


if __name__ == "__main__":
    loss_mse()
