import tensorflow as tf
import numpy as np
import pandas as pd

class DotModel(tf.keras.Model):
    def __init__(self):
        super(DotModel, self).__init__()
        self.d1 = tf.keras.layers.Dense(4, activation="relu", kernel_regularizer=tf.keras.regularizers.L2())
        self.d2 = tf.keras.layers.Dense(8, activation="relu", kernel_regularizer=tf.keras.regularizers.L2())
        self.d3 = tf.keras.layers.Dense(16, activation="relu", kernel_regularizer=tf.keras.regularizers.L2())
        self.d4 = tf.keras.layers.Dense(1, activation="relu", kernel_regularizer=tf.keras.regularizers.L2())
    def call(self, x):
        y = self.d1(x)
        y = self.d2(y)
        y = self.d3(y)
        y = self.d4(y)
        return y

if __name__ == "__main__":
    data = pd.read_csv("dot.csv")
    x_data = np.array(data[["x1", "x2"]])
    y_data = np.array(data["y_c"]).reshape(-1, 1)
    model = DotModel()
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01), loss="mse", metrics=["accuracy"])
    model.fit(x_data, y_data, batch_size=32, epochs=1500, validation_split=0.1, validation_freq=50)
    model.summary()

