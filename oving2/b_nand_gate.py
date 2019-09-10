#%% setup

import tensorflow as tf

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

from common import *
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('TkAgg')


def f_real_xz(x, z, W, b):
    return sigmoid(x*W[0] + z*W[1] + b)


class Model:
    def __init__(self):
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)
        self.W = tf.Variable([[0.0], [0.0]])
        self.b = tf.Variable([0.])
        self.logits = tf.matmul(self.x, self.W)+self.b
        self.loss = tf.losses.sigmoid_cross_entropy(self.y, self.logits)
        self.minimize_op = tf.train.GradientDescentOptimizer(100).minimize(self.loss)

    def train(self, session, x_t, y_t, times=1000):
        for epoch in range(times):
            session.run(self.minimize_op, {self.x: x_t, self.y: y_t})
        return session.run([self.W, self.b, self.loss], {self.x: x_t, self.y: y_t})


x_train = []
y_train = []
for line in open('data_b.csv'):
    x, z, y = line.split(',')
    x_train.append([float(x), float(z)])
    y_train.append(float(y))

x_train = np.matrix(x_train)
y_train = np.matrix(y_train).T

model = Model()
tf_session = tf.Session()
tf_session.run(tf.global_variables_initializer())
tf_session.run(tf.local_variables_initializer())


#%% train

W, b, loss = model.train(tf_session, x_train, y_train, 10000)
print("W = %s, b = %s, loss = %s" % (W, b, loss))

#%% graph


f_temp = lambda x, z: f_real_xz(x, z, W, b)
graph3d(x_train, y_train, f_temp)

#%%close
tf_session.close()
