import numpy as np
from numpy import linalg
from scipy import optimize as op
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def getPlane(points):
    iter = 100
    n = points.shape[0]
    maxinliers = 0
    bestplane = np.array([])
    threold = 0.003
    T = 0.9 * n


    for i in range(iter):
        A = np.zeros([3,3],dtype=np.float32)
        ran = [x for x in range(0, n)]
        random.shuffle(ran)
        for j in range(3):
            A[j,:] = points[ran[j],:]
        plane = get_nice_plane(A)
        best_point = np.array([])
        inliers=0
        for j in range(n):
            if abs(plane[0]*points[j,0]+plane[1]*points[j,1]+plane[2]-points[j,2])<threold:
                inliers = inliers+1
                best_point = np.append(best_point,points[j,:])
        if inliers>T:
            best_point = best_point.reshape(-1,3)
            bestplane = get_nice_plane(best_point)
            break
        if inliers>maxinliers:
            maxinliers=inliers
            bestplane = plane
    drawPlane(bestplane,points)
    return bestplane
def loss_function(x,point):
    error =  point[:,0]*x[0]+point[:,1]*x[1]-point[:,2]+x[2]
    return error


def get_nice_plane(point):
    root = op.leastsq(loss_function, [1,1,1],args=(point))
    return root[0]

def drawPlane(plane,point):
    maxP = np.max(point,axis=0)
    minP = np.min(point,axis=0)
    xx = np.linspace(minP[0],maxP[0] , 100)
    yy = np.linspace(minP[1],maxP[1] , 100)
    X, Y = np.meshgrid(xx, yy)
    Z = plane[0]*X+plane[1]*Y+plane[2]
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot_surface(X, Y, Z, cmap='rainbow')
    ax.scatter3D(point[:,0], point[:,1], point[:,2], cmap='Blues')
    plt.show()



