import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets

def iris_train(epoch,lr):
    x_data=datasets.load_iris().data
    y_data=datasets.load_iris().target
    np.random.seed(100)
    np.random.shuffle(x_data)
    np.random.seed(100)
    np.random.shuffle(y_data)
    x_train=x_data[:-30]
    x_test=x_data[-30:]
    y_train=y_data[:-30]
    y_test=y_data[-30:]
    x_train=tf.cast(x_train,dtype=tf.float32)
    x_test=tf.cast(x_test,dtype=tf.float32)
    train = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(30)
    test=tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(30)
    w = tf.Variable(tf.random.truncated_normal([4,3],stddev=1))
    b = tf.Variable(tf.random.truncated_normal([3],stddev=1))
    loss_all=0
    train_loss=[]
    test_acc=[]
    for epochs in range(epoch):
        for step, (x_trains, y_trains) in enumerate(train):
            with tf.GradientTape() as tape:
                y = tf.matmul(x_trains, w)+b
                y = tf.nn.softmax(y)
                y_ = tf.one_hot(y_trains, depth=3)
                loss = tf.reduce_mean(tf.square(y-y_))
                loss_all += loss.numpy()
            lr_=lr*0.99**(epochs/1)
            grad = tape.gradient(loss, [w, b])
            w.assign_sub(lr_ * grad[0])
            b.assign_sub(lr_ * grad[1])
            print("epoch:%d step:%d loss:%f" % (epochs, step, loss))
        train_loss.append(loss_all/4)
        loss_all=0
        total_correct=0
        total_number=0
        for step,(x_test,y_test) in enumerate(test):
            y=tf.matmul(x_test,w)+b
            y=tf.nn.softmax(y)
            pred=tf.argmax(y,axis=1)
            pred=tf.cast(pred,dtype=y_test.dtype)
            correct=tf.cast(tf.equal(pred,y_test),dtype=tf.int32)
            correct=tf.reduce_sum(correct)
            total_correct+=int(correct)
            total_number+=x_test.shape[0]
        test_acc.append(float(total_correct/total_number))
        print("%d %f" % (epochs,total_correct/total_number))
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(train_loss,label="loss")
    plt.show()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.plot(test_acc, label="acc")
    plt.show()

if __name__ == "__main__":
    iris_train(500,0.5)