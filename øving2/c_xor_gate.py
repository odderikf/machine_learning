#%% setup

import tensorflow as tf

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

from common import *
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('TkAgg')

#import random

def f_part(x, W, b):
    return sigmoid(x * W + b)


def f_real(x, W1, W2, b1, b2):
    h = f_part(np.mat(x), np.mat(W1), b1)
    return f_part(h, W2, b2)

# Makes the y = f(x,z) mesh grid corresponding to z and x mesh grids
def f_grid(x_mesh, z_mesh, W1, W2, b1, b2):
    y_mesh = np.empty(np.shape(x_mesh))

    for i in range(0, x_mesh.shape[0]):
        for j in range(0, x_mesh.shape[1]):
            v = f_real([[x_mesh[i, j], z_mesh[i, j]]], W1, W2, b1, b2)
            y_mesh[i, j] = v

    return y_mesh


# Finding the magic start numbers that converge to the right solution
#random.seed()
#random_nums = [random.random()*2 - 1 for _ in range(9)]
#print('rands:', random_nums)

# Magic numbers from successful run
random_nums = [0.970914018468493, 0.8614498859618396, 0.6441907794405679, 0.12523876002271872, 0.23437941719162003, 0.364447426837758, -0.8798999218739767, -0.9098780339454522, -0.3820887237650934]


class Model:
    def __init__(self):
        self.x1 = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)
        self.W1 = tf.Variable([[random_nums.pop(), random_nums.pop()], [random_nums.pop(), random_nums.pop()]])  # 2x1 * 2x2 = 2x1
        self.W2 = tf.Variable([[random_nums.pop()], [random_nums.pop()]])  # 2x1 * 1x2 = 1x1
        self.b1 = tf.Variable([random_nums.pop(), random_nums.pop()]) # matches x1
        self.b2 = tf.Variable([random_nums.pop()]) # matches x2
        self.logits1 = tf.matmul(self.x1, self.W1) + self.b1  # 2x1
        self.x2 = tf.sigmoid(self.logits1)  # 2x1
        self.logits2 = tf.matmul(self.x2, self.W2)+self.b2  # 1x1
        self.f = tf.sigmoid(self.logits2)
        self.loss = tf.losses.sigmoid_cross_entropy(self.y, self.logits2)
        self.minimize_op = tf.train.GradientDescentOptimizer(10).minimize(self.loss)

    def train(self, session, x_t, y_t, times=10000):
        for epoch in range(times):
            session.run(self.minimize_op, {self.x1: x_t, self.y: y_t})
        return session.run([self.W1, self.b1, self.W2, self.b2, self.loss], {self.x1: x_t, self.y: y_t})


x_train = []
y_train = []
for line in open('data_c.csv'):
    x, z, y = line.split(',')
    x_train.append([float(x), float(z)])
    y_train.append(float(y))

x_train = np.matrix(x_train)
y_train = np.matrix(y_train).T

model = Model()
tf_session = tf.Session()
tf_session.run(tf.global_variables_initializer())
tf_session.run(tf.local_variables_initializer())
writer = tf.summary.FileWriter('logs', tf_session.graph)


#%% train

W1, b1, W2, b2, loss = model.train(tf_session, x_train, y_train, 10000)
print("W1 = %s, b1 = %s, W2 = %s, b2 = %s, loss = %s" % (W1, b1, W2, b2, loss))

#%% graph

f_temp = lambda x, z: f_grid(x, z, W1, W2, b1, b2)
graph3d(x_train, y_train, f_temp)

#%%close
writer.close()
tf_session.close()
