import numpy as np

import matplotlib
from matplotlib import pyplot as plt


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def softmax(Z):
    e_x = np.exp(Z - np.max(Z))
    return e_x / e_x.sum()


def graph2d(x, y, f):
    plt.scatter(x.T.tolist()[0], y.T.tolist()[0], c='red', alpha=0.5)
    min_x = float(min(min(x.tolist())))
    max_x = float(max(max(x.tolist())))
    dx = max_x-min_x
    x_plt = np.linspace(min_x-0.1*dx, max_x+0.1*dx, num=400)
    y_plt = f(x_plt)
    plt.plot(x_plt, y_plt[0], c='blue')
    plt.show()


def graph3d(X, y, f):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')

    x_scatt, z_scatt = [i.tolist()[0] for i in X.T]
    y_scatt = y.T.tolist()[0]
    ax.scatter3D(x_scatt, z_scatt, y_scatt,  c='r')

    x_min = min(x_scatt)
    x_max = max(x_scatt)
    dx = x_max - x_min
    z_min = min(z_scatt)
    z_max = max(z_scatt)
    dz = z_max - z_min

    x_lin = np.linspace(x_min - 0.1*dx, x_max + 0.1*dx, num=30)
    z_lin = np.linspace(z_min - 0.1*dz, z_max + 0.1*dz, num=30)
    x_mesh, z_mesh = np.meshgrid(x_lin, z_lin)
    y_mesh = f(x_mesh, z_mesh)  # often works withut mesh unravelling, but in c mesh unravelling was needed, and many tears were shed
    ax.plot_surface(x_mesh, z_mesh, y_mesh, alpha=0.8)
    plt.show()


