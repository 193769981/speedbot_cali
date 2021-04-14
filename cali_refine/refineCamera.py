# _*_ coding:utf-8 _*_

import numpy as np
import sys
ros_cv2_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_cv2_path in sys.path:
    sys.path.remove(ros_cv2_path)
    import cv2
    sys.path.append(ros_cv2_path)
else:
    import cv2
import transforms3d
from scipy import optimize as opt

def refinall_all_param(A, k, W, real_coor, pic_coor):
    '''
    :param A: 内参
    :param k: 畸变参数
    :param W: 外参
    :param real_coor: 标定板坐标
    :param pic_coor: 像素坐标
    :return:
    '''
    discoff_len = k.shape[1]
    P_init = compose_paramter_vector(A, k, W)
    error = lossfunction(P_init, discoff_len, pic_coor, real_coor)

    # solver2 = opt.least_squares(lossfunction,P_init,args=(discoff_len,pic_coor,real_coor),method="lm",jac='3-point')
    #
    # error = lossfunction(solver2.x, discoff_len, pic_coor, real_coor)
    options = {'col_deriv': 0, 'xtol': 0.001, 'ftol': 1.49012e-15, 'gtol': 0.0, 'maxiter': 10000000,
               'eps': 0.0, 'factor': 100, 'diag': None}
    solver = opt.root(lossfunction, P_init, args=(discoff_len, pic_coor, real_coor), method="lm", tol=None, options=options)
    P = solver.x
    error = lossfunction(P, discoff_len, pic_coor, real_coor)
    return decompose_paramter_vector(P, discoff_len)


def lossfunction(P, discoff_len, pic_coor, real_coor):
    A, discoff, w = decompose_paramter_vector(P, discoff_len)

    errors = np.array([])
    for i in range(len(w)):
        rvec = w[i][:3, :3]
        tvec = w[i][:3, 3]
        real_coor_temp = np.append(real_coor[i], np.zeros([real_coor[i].shape[0], 1]), 1)
        rvec = cv2.Rodrigues(rvec)[0]
        imagePoints, jacobian = cv2.projectPoints(real_coor_temp, rvec, tvec, A, discoff)
        imagePoints = imagePoints.reshape([-1, 2])
        error = imagePoints - pic_coor[i]
        errors = np.append(errors, error)
    return np.abs(errors)#.flatten()





def compose_paramter_vector(A, k, W):
    alpha = np.array([A[0, 0], A[1, 1], A[0, 2], A[1, 2]])
    alpha = np.append(alpha,k)
    P = alpha
    for i in range(len(W)):
        R, t = (W[i])[:3, :3], (W[i])[:3, 3]
        # 旋转矩阵转换为一维向量形式
        zrou = transforms3d.quaternions.mat2quat(R)

        w = np.append(zrou, t)
        P = np.append(P, w)
    return P

def decompose_paramter_vector(P,lengthdiscoff):
    [alpha, gamma, uc, vc] = P[0:4]
    discoff = P[4:4+lengthdiscoff]
    A = np.array([[alpha, 0, uc],
                  [0, gamma, vc],
                  [0, 0, 1]])

    W = []
    M = (len(P) - 4-lengthdiscoff)/7

    for i in range(int(M)):
        m = 4+lengthdiscoff + 7 * i
        zrou = P[m:m + 4]
        t = (P[m + 4:m + 7]).reshape(3, -1)

        # 将旋转矩阵一维向量形式还原为矩阵形式
        R = transforms3d.quaternions.quat2mat(zrou)

        # 依次拼接每幅图的外参
        w = np.append(R, t, axis=1)
        w = np.append(w,np.array([[0,0,0,1]]),0)
        W.append(w)

    return A, discoff, W