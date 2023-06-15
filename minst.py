import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.L2()),
        tf.keras.layers.Dense(10, activation="softmax", kernel_regularizer=tf.keras.regularizers.L2())
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["sparse_categorical_accuracy"])
    model.fit(x_train, y_train, batch_size=32, epochs=8, validation_data=(x_test, y_test), validation_freq=1)
    model.summary()