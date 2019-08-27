#%% setup

import tensorflow as tf

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

from common import *


def f_real(x, W, b):
    return sigmoid(x*W+b)


class Model:
    def __init__(self):
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)
        self.W = tf.Variable([[0.]])
        self.b = tf.Variable([0.])
        self.logits = tf.matmul(self.x, self.W)+self.b
        self.loss = tf.losses.sigmoid_cross_entropy(self.y, self.logits)
        self.minimize_op = tf.train.GradientDescentOptimizer(100).minimize(self.loss)

    def train(self, session, x_t, y_t, times=1000):
        for epoch in range(times):
            session.run(self.minimize_op, {self.x: x_t, self.y: y_t})
        return session.run([self.W, self.b, self.loss], {self.x: x_t, self.y: y_t})


x_not = []
y_not = []
for line in open('data_a.csv'):
    x, y = line.split(',')
    x_not.append(float(x))
    y_not.append(float(y))

x_not = np.matrix(x_not).T
y_not = np.matrix(y_not).T

model = Model()
tf_session = tf.Session()
tf_session.run(tf.global_variables_initializer())
tf_session.run(tf.local_variables_initializer())


#%% train

W, b, loss = model.train(tf_session, x_not, y_not, 20000)
print("W = %s, b = %s, loss = %s" % (W, b, loss))

#%% graph


f_temp = lambda x: f_real(x, W, b)
graph2d(x_not, y_not, f_temp)

#%%close
tf_session.close()
