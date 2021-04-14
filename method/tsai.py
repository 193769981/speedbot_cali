# -*- coding: utf-8 -*-
import numpy as np
import sys
ros_cv2_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_cv2_path in sys.path:
    sys.path.remove(ros_cv2_path)
    import cv2
    sys.path.append(ros_cv2_path)
else:
    import cv2
import math


def skew(N):
    return np.array([
        [0, -N[2][0], N[1][0]],
        [N[2][0], 0, -N[0][0]],
        [-N[1][0], N[0][0], 0]
    ])


def calibration(A, B):
    '''
    使用Tsai算法进行标定处理
    :param A: AX=XB,来自handineye.motion_axxb或者handtoeye.motion_axxb
    :param B: AX=XB,来自handineye.motion_axxb或者handtoeye.motion_axxb
    :return: H：若使用handineye，为机器臂末端到相机的变换，使用handtoeye是机器臂基底到相机的变换
    '''
    V = np.array([])
    b = np.array([])
    for i in range(len(A)):
        Hbij = B[i]
        Haij = A[i]
        Rbij = Hbij[0:3,0:3]
        Raij = Haij[0:3,0:3]
    #使用Rodrigues方法转化为旋转向量
        rbij = cv2.Rodrigues(Rbij)[0]
        raij = cv2.Rodrigues(Raij)[0]
    #归一化
        thetabij = np.linalg.norm(rbij)
        thetaaij = np.linalg.norm(raij)
        Nbij = rbij/thetabij
        Naij = raij/thetabij
    #利用修正的Rodrigues表示姿态
        Pbij = 2* math.sin(thetabij/2)*Nbij
        Paij = 2* math.sin(thetaaij/2)*Naij
        bb = np.array([Pbij-Paij])
        v = skew(Paij + Pbij)
        if not (np.isnan(b).any() or np.isnan(v).any()):
            b = np.append(b,bb)
            V = np.append(V,v)
    V = V.reshape(-1,3)
    b = b.reshape(-1,1)
    Px_temp = np.linalg.lstsq(V,b, rcond=-1)[0]
    Px = 2 * Px_temp / math.sqrt(1 + np.linalg.norm(Px_temp) ** 2)
    # 计算旋转矩阵Rx
    Px_norm = np.linalg.norm(Px)
    Rx = (1 - Px_norm ** 2 / 2) * np.identity(3) + 0.5 * (
                np.dot(Px, Px.T) + math.sqrt(4 - Px_norm ** 2) * skew(Px))
    # 计算平移向量Tx
    M = []
    AMatrix = np.array([])
    BMatrix = np.array([])
    for i in range(len(A)):
        Hcij = B[i]
        Haij = A[i]
        Raij = Haij[0:3, 0:3]
        Tcij = Hcij[0:3, 3]
        Taij = Haij[0:3, 3]
        AA = Raij - np.identity(3)
        AMatrix = np.append(AMatrix, np.array([AA]))
        BB = np.dot(Rx, np.transpose([Tcij])) - np.transpose([Taij])
        BMatrix = np.append(BMatrix, np.array([BB]))
        M.append(np.array([AA[0, 0], AA[0, 1], AA[0, 2], -BB[0, 0]]))
        M.append(np.array([AA[1, 0], AA[1, 1], AA[1, 2], -BB[1, 0]]))
        M.append(np.array([AA[2, 0], AA[2, 1], AA[2, 2], -BB[2, 0]]))
    AMatrix = AMatrix.reshape(-1, 3)
    BMatrix = BMatrix.reshape(-1, 1)
    Tx = np.linalg.lstsq(AMatrix, BMatrix, rcond=-1)[0]
    Hx = np.append(Rx, Tx, axis=1)
    Hx = np.append(Hx, [[0, 0, 0, 1]], axis=0)

    # error = np.array([])
    # for i in range(len(A)):
    #     error=np.append(error,np.sum(np.dot(A[i],Hx)-np.dot(Hx,B[i])))
    #     T = np.dot(A[i],Hx)-np.dot(Hx,B[i])
    #     for j in range(3):
    #         if T[j,3]>0.01 or T[j,3]<-0.01:
    #             print(i)
    #     # print np.dot(A[i],Hx)-np.dot(Hx,B[i])
    # print(error)

    return Hx