# _*_ coding:utf-8 _*_
# @time: 2020/9/29 上午9:56
# @author: 张新新
# @email: 1262981714@qq.com
import numpy as np
import transforms3d
from scipy import optimize as op
import cv2
from matplotlib import pyplot as plt
def get_nice_plane(point):
    def loss_function(x, point):
        error = point[:, 0] * x[0] + point[:, 1] * x[1] - point[:, 2] + x[2]
        return error

    root = op.leastsq(loss_function, np.array([1, 1, 1]), args=(point))
    return root[0]


def drawPlane(plane, point):
    maxP = np.max(point, axis=0)
    minP = np.min(point, axis=0)
    xx = np.linspace(minP[0], maxP[0], 100)
    yy = np.linspace(minP[1], maxP[1], 100)
    X, Y = np.meshgrid(xx, yy)
    Z = plane[0] * X + plane[1] * Y + plane[2]
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot_surface(X, Y, Z, cmap='rainbow')
    ax.scatter3D(point[:, 0], point[:, 1], point[:, 2], cmap='Blues')
    plt.show()