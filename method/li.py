# -*- coding: utf-8 -*-
import numpy as np
import math
from scipy import optimize as op

def calibration(motionAs, motionBs):
    '''
    使用li算法解AX=YB的方程，求解手眼标定
    :param motionAs: list<array<4*4>> A 来自handineye.motion_axyb或者是handtoeye.motion_axyb
    :param motionBs: list<array<4*4>> B 来自handineye.motion_axyb或者是handtoeye.motion_axyb
    :return: X:handineye中为机器基底到标定板的变换 handtoeye:相机到底座的变换
            Y:handineye中为机器臂末端到相机的变换 handtoeye:标定板到机械臂末端的变换
    '''
    T = np.zeros([9, 9])
    n = len(motionAs)
    A = np.zeros([12*n, 24])
    b = np.zeros([12*n, 1])
    for i in range(n):
        # AA为机器人位姿，4×4的矩阵
        AA = motionAs[i]
        # BB是相机位姿，4×4的矩阵
        BB = motionBs[i]
        # 3×3的机器人旋转矩阵
        Ra = AA[:3, :3]
        # ta为机器人平移向量
        ta = AA[:3, 3]
        # 3×3的相机旋转矩阵
        Rb = BB[:3, :3]
        # tb为相机平移向量
        tb = BB[:3, 3]
        A[12*i:12*i+9, 0:18] = np.append(np.kron(Ra, np.identity(3)), np.kron(-np.identity(3), Rb.T), 1)
        # t = np.append(np.kron(np.identity(3),tb.T),np.append(-Ra,np.identity(3),1),1)
        A[12*i+9:12*(i+1), 9:24] = np.append(np.kron(np.identity(3), tb.T), np.append(-Ra, np.identity(3), 1), 1)
        b[12*i+9:12*(i+1)] = np.array([ta]).T
    x = np.linalg.lstsq(A, b, rcond=-1)[0]

    X = np.reshape(x[0:9, :], [3, 3], order="F").T
    U, S, VT = np.linalg.svd(X)
    X = np.dot(U, VT)
    if np.linalg.det(X) < 0:
        X = np.dot(U, np.dot(np.diag([1, 1, -1]), VT))
    X = np.append(np.append(X, x[18:21, :], 1), np.array([[0, 0, 0, 1]]), 0)

    Y = np.reshape(x[9:18, :], [3, 3], order="F").T
    U, S, VT = np.linalg.svd(Y)
    Y = np.dot(U, VT)
    if np.linalg.det(Y) < 0:
        Y = np.dot(U, np.dot(np.diag([1, 1, -1]), VT))
    Y = np.append(np.append(Y, x[21:24, :], 1), np.array([[0, 0, 0, 1]]), 0)
    return X, Y