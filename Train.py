import tensorflow as tf


def train_x(w, lr, epoch):
    for epochs in range(epoch):
        with tf.GradientTape() as tape:
            loss = tf.square(w)
            grad = tape.gradient(loss, w)
        w.assign_sub(lr * grad)

        print("%d %f %f" % (epochs, w.numpy(), loss))


if __name__ == "__main__":
    w = tf.Variable(tf.constant(3, dtype=tf.float32))
    lr = 0.2
    epoch = 40
    train_x(w, lr, epoch)
