
import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


x_train = []  # length
y_train = []  # weight

with open('data/del_a.csv') as file:
    for line in file:
        if '#' not in line:
            x_i, y_i = line.split(',')
            x_train.append([float(x_i)])
            y_train.append([float(y_i)])

x_train = np.mat(x_train)
y_train = np.mat(y_train)

x_tf = tf.placeholder(tf.float32)
y_tf = tf.placeholder(tf.float32)
W_tf = tf.Variable([[0.]])
b_tf = tf.Variable([[0.]])

f = tf.matmul(x_tf, W_tf) + b_tf

loss_tf = tf.reduce_mean(tf.square(f - y_tf))

minimize_op = tf.train.GradientDescentOptimizer(0.0001).minimize(loss_tf)

session = tf.Session()

session.run(tf.global_variables_initializer())

for epoch in range(140000):
    session.run(minimize_op, {x_tf: x_train, y_tf: y_train})

W, b, loss = session.run([W_tf, b_tf, loss_tf], {x_tf: x_train, y_tf: y_train})
print("W = %s, b = %s, loss = %s" % (W, b, loss))

session.close()

plt.scatter(x_train.T.tolist()[0], y_train.T.tolist()[0])
x_plt = np.linspace(float(min(x_train)), float(max(x_train)))

plt.plot(x_plt, float(W)*x_plt+float(b), 'r')
plt.show()
