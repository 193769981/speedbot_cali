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


def refine_extrinsic_lm(A, k, W, real_coor, pic_coor):
    '''
    利用A,k,real_coor,pic_coor对W（相机外参）进行优化，方法为lm非线性最小二乘优化
    :param A:相机内参
    :param k: 畸变系数
    :param W: 多张图片对应的相机姿态一维表示的拼接
    :param real_coor: board上角点在世界坐标系下的坐标(多张照片角点的集合)
    :param pic_coor: board上对应角点在图片中的坐标(多张照片角点的集合)
    :return:优化后的相机外参
    '''
    P_init = compose_paramter_vector(W)
    # 解lossfunction=0方程的根：P
    solver = opt.root(lossfunction, P_init, args=(A, k, pic_coor, real_coor), method="lm")
    P = solver.x
    return decompose_paramter_vector(P)


def lossfunction(P, A, discoff, pic_coor, real_coor):
    '''
    计算在像素坐标系下的损失
    :param P:四元数+平移向量
    :param A:相机内参
    :param discoff:畸变系数
    :param pic_coor:图片上的坐标
    :param real_coor:世界坐标系（标定板）上的坐标
    :return: 投影到像素坐标上来计算损失，并将多个损失展平
    '''
    w = decompose_paramter_vector(P)

    errors = np.array([])
    for i in range(len(w)):
        rvec = w[i][:3, :3]
        tvec = w[i][:3, 3]
        # 最后一列加0，相当于Z轴的坐标变为0
        real_coor_temp = np.append(real_coor[i], np.zeros([real_coor[i].shape[0], 1]), 1)
        imagePoints, jacobian = cv2.projectPoints(real_coor_temp, rvec, tvec, A, discoff)
        imagePoints = imagePoints.reshape([-1, 2])
        error = imagePoints - pic_coor[i]
        errors = np.append(errors, error)
    return errors.flatten()


def compose_paramter_vector(W):
    '''
    :param W: 单应性矩阵
    :return:将单应性矩阵W转化为“四元数+平移向量”的形式
    '''
    P = np.array([])
    for i in range(len(W)):
        R, t = (W[i])[:3, :3], (W[i])[:3, 3]
        # 旋转矩阵转换为四元数
        zrou = transforms3d.quaternions.mat2quat(R)
        w = np.append(zrou, t)
        P = np.append(P, w)
    return P


def decompose_paramter_vector(P):
    '''
    :param P:"四元数+平移向量"的一维表示形式，一共7个数
    :return: 单应性矩阵
    '''
    W = []
    M = len(P) / 7

    # 用7个数表示一个矩阵，前4个为四元数，后3个为平移向量
    for i in range(int(M)):
        m = 7 * i
        zrou = P[m:m + 4]
        t = (P[m + 4:m + 7]).reshape(3, -1)
        # 将旋转矩阵一维向量形式还原为矩阵形式
        R = transforms3d.quaternions.quat2mat(zrou)
        # 依次拼接每幅图的外参w，最终将w都添加到W中
        w = np.append(R, t, axis=1)
        w = np.append(w, np.array([[0, 0, 0, 1]]), 0)
        W.append(w)
    return W