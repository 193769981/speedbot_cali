# -*- coding:utf-8 -*-
import numpy as np
from utils import tranformUtils as utils


def calibration(motionAs, motionBs):
    '''
    使用对偶四元数进行标定
    :param motionAs: AX=XB,来自handineye.motion_axxb或者handtoeye.motion_axxb
    :param motionBs:  AX=XB,来自handineye.motion_axxb或者handtoeye.motion_axxb
    :return: 若使用handineye，为机器臂末端到相机的变换，使用handtoeye是机器臂基底到相机的变换
    '''
    size = len(motionAs)
    T = []
    for j in range(size):
        motionA = motionAs[j]
        motionB = motionBs[j]
        qa, qa_prime = utils.rot2dualquat(motionA)
        qb, qb_prime = utils.rot2dualquat(motionB)
        p = utils.get_sub_mat(qa, qa_prime, qb, qb_prime)
        if not np.isnan(p).any():
            T.append(p)
    T = np.concatenate(T)
    U, s, V = np.linalg.svd(T)
    idx1, idx2 = np.argsort(s)[:2].tolist()
    v7 = V[idx1]
    v8 = V[idx2]

    u1 = v7[:4]
    v1 = v7[4:]
    u2 = v8[:4]
    v2 = v8[4:]

    a = np.dot(u1, v1)
    b = np.dot(u1, v2) + np.dot(u2, v1)
    c = np.dot(u2, v2)

    s1 = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    s2 = (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)

    x1 = s1 ** 2 * np.dot(u1, u1) + 2 * s1 * np.dot(u1, u2) + np.dot(u2, u2)
    x2 = s2 ** 2 * np.dot(u1, u1) + 2 * s2 * np.dot(u1, u2) + np.dot(u2, u2)
    (x, s) = (x1, s1) if x1 >= x2 else (x2, s2)

    lambda2 = np.sqrt(1 / x)
    lambda1 = s * lambda2

    q = lambda1 * u1 + lambda2 * u2
    q_ = lambda1 * v1 + lambda2 * v2

    H = np.append(utils.dualquat2r_t(q, q_), np.array([[0, 0, 0, 1]]), 0)
    error = np.array([])
    for i in range(len(motionAs)):
        error = np.append(error, np.sum(np.dot(motionAs[i], H) - np.dot(H, motionBs[i])))

    return H