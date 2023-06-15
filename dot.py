import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def dot_train(epochs, lr, e):
    data = pd.read_csv("dot.csv")
    x_data = np.array(data[["x1", "x2"]])
    y_data = np.array(data[["y_c"]]).reshape(-1, 1)
    y_c = [['red' if y else 'blue']for y in y_data]
    x_train = tf.cast(x_data, dtype=tf.float32)
    y_train = tf.cast(y_data, dtype=tf.float32)
    train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
    w1 = tf.Variable(tf.random.truncated_normal([2, 16]))
    b1 = tf.Variable(tf.random.truncated_normal([16]))
    w2 = tf.Variable(tf.random.truncated_normal([16, 8]))
    b2 = tf.Variable(tf.random.truncated_normal([8]))
    w3 = tf.Variable(tf.random.truncated_normal([8, 1]))
    b3 = tf.Variable(tf.random.truncated_normal([1]))
    final_epoch, final_loss = 0, 0.0
    for epoch in range(epochs):
        for step, (x_trains, y_trains) in enumerate(train):
            with tf.GradientTape() as tape:
                h1 = tf.matmul(x_trains, w1) + b1
                h1 = tf.nn.tanh(h1)
                h1 = tf.matmul(h1, w2) + b2
                h1 = tf.nn.tanh(h1)
                y = tf.matmul(h1, w3) + b3
                loss = tf.reduce_mean(tf.square(y, y_trains))
                reg_loss = [tf.nn.l2_loss(w1), tf.nn.l2_loss(w2),tf.nn.l2_loss(w3)]
                loss_w = tf.reduce_sum(reg_loss)
                loss_ = loss + 0.03*loss_w
                grad = tape.gradient(loss_, [w1, b1, w2, b2, w3, b3])
                lr_ = lr
                w1.assign_sub(lr_ * grad[0])
                b1.assign_sub(lr_ * grad[1])
                w2.assign_sub(lr_ * grad[2])
                b2.assign_sub(lr_ * grad[3])
                w3.assign_sub(lr_ * grad[4])
                b3.assign_sub(lr_ * grad[5])
        print("epoch:%d, loss:%f" % (epoch, loss))
        final_epoch = epoch
        final_loss = loss
        if loss < e:
            break
    print("final epoch:%d, loss:%f" % (final_epoch, final_loss))
    x, y = np.mgrid[-3:3:.1, -3:3:.1]
    grid = np.c_[x.ravel(), y.ravel()]
    grid = tf.cast(grid, tf.float32)
    pred = []
    for x_test in grid:
        h1 = tf.matmul([x_test], w1) + b1
        h1 = tf.nn.tanh(h1)
        h1 = tf.matmul(h1, w2) + b2
        h1 = tf.nn.tanh(h1)
        pred.append(tf.matmul(h1, w3) + b3)
    pred = np.array(pred).reshape(x.shape)
    plt.scatter(x_train[:, 0], x_train[:, 1], color=np.squeeze(y_c))
    plt.contour(x, y, pred, levels=[5])
    plt.show()

if __name__ == "__main__":
    dot_train(1500, 0.005, 0.001)
