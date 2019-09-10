#%% setup
import numpy as np
import tensorflow as tf

import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('TkAgg')

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


x_train = []  # length
y_train = []  # weight

with open('data/del_b.csv') as file:
    for line in file:
        if '#' not in line:
            y_i, x_1_i, x_2_i = line.split(',')
            x_train.append([float(x_1_i), float(x_2_i)])
            y_train.append([float(y_i)])

x_train = np.mat(x_train)
y_train = np.mat(y_train)

x_tf = tf.placeholder(tf.float32)
y_tf = tf.placeholder(tf.float32)
W_tf = tf.Variable([[0.], [0.]])
b_tf = tf.Variable([[0.]])

f = tf.matmul(x_tf, W_tf) + b_tf

loss_tf = tf.reduce_mean(tf.square(f - y_tf))

minimize_op = tf.train.GradientDescentOptimizer(0.0001).minimize(loss_tf)

session = tf.Session()

session.run(tf.global_variables_initializer())

#%% train
for epoch in range(100000):
    session.run(minimize_op, {x_tf: x_train, y_tf: y_train})

W, b, loss = session.run([W_tf, b_tf, loss_tf], {x_tf: x_train, y_tf: y_train})
print("W = %s, b = %s, loss = %s" % (W, b, loss))

#%% graph
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

x_scatt, z_scatt = [i.tolist()[0] for i in x_train.T]
y_scatt = y_train.T.tolist()[0]
ax.scatter3D(x_scatt, y_scatt, z_scatt, c='r', alpha=0.5)

x_lin = np.linspace(float(min(x_scatt)), float(max(x_scatt)))
z_lin = np.linspace(float(min(z_scatt)), float(max(z_scatt)))
x_mesh, z_mesh = np.meshgrid(x_lin, z_lin)
X_mat = np.mat([x_mesh.tolist()[0], z_mesh.tolist()[0]]).T
y_plt = x_mesh*W[0] + z_mesh*W[1] + b
ax.plot_surface(x_mesh, y_plt, z_mesh, alpha=0.3, color='blue')
plt.show()

#%% close
session.close()
