#%% setup

import tensorflow as tf

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

from common import *

import datetime

(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = tf.keras.datasets.mnist.load_data()

x_train = x_train_raw.reshape((x_train_raw.shape[0], x_train_raw.shape[1] * x_train_raw.shape[2]))
y_train = [[0 for _ in range(10)] for _ in range(y_train_raw.shape[0])]
for raw, new in zip(y_train_raw, y_train):
    new[raw] = 1


x_test = x_test_raw.reshape((x_test_raw.shape[0], x_test_raw.shape[1] * x_test_raw.shape[2]))
y_test = [[0 for _ in range(10)] for _ in range(y_test_raw.shape[0])]
for raw, new in zip(y_test_raw, y_test):
    new[raw] = 1


class Model:
    def __init__(self):
        self.x = tf.placeholder(tf.float64)  # 1x784
        self.y = tf.placeholder(tf.float64)  # 1x10
        self.W = tf.Variable(np.ndarray([784, 10]))  # 1x784 x 784x10 = 1x10
        self.b = tf.Variable(np.ndarray([1, 10]))  # matches y
        self.logits = tf.matmul(self.x, self.W) + self.b  # 1x10
        self.f = tf.nn.softmax(self.logits)
        self.loss = tf.losses.softmax_cross_entropy(self.y, self.logits)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.f, 1), tf.argmax(self.y, 1)), tf.float32))
        self.minimize_op = tf.train.GradientDescentOptimizer(100000000000.).minimize(self.loss)

    def train(self, session, x_t, y_t, times=10000):
        for epoch in range(times):
            session.run(self.minimize_op, {self.x: x_t, self.y: y_t})
        return session.run([self.W, self.b, self.loss, self.accuracy], {self.x: x_t, self.y: y_t})


model = Model()
tf_session = tf.Session()
tf_session.run(tf.global_variables_initializer())
tf_session.run(tf.local_variables_initializer())


#%% train

W, b, loss, accuracy = model.train(tf_session, x_train, y_train, 200)
print("W = %s, b = %s, loss = %s, accuracy = %s" % (W, b, loss, accuracy))

#%% save W to image

now = datetime.datetime.now()
dir = f'images/accuracy_{accuracy}_date_{now.year}.{now.month}.{now.day}_{now.hour}.{now.minute}'
mkdir_p(dir)
for i in range(10):
    W_im = W[:, i].reshape([28, 28])
    plt.imsave(
        f'{dir}/result_{i}.png',
        W_im
    )
#%%close
tf_session.close()
