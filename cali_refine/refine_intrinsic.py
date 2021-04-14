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
import math
from utils import plane_ransac


def refinall_intrinsic(A, k, W, real_coor, pic_coor):
    discoff_len = k.shape[1]
    P_init = compose_paramter_vector(A, k)


    solver = opt.root(lossfunction,P_init,args=(discoff_len,W,pic_coor,real_coor),method="lm")
    P = solver.x
    return decompose_paramter_vector(P,discoff_len)


def lossfunction(P,discoff_len,w,pic_coor,real_coor):
    A,discoff= decompose_paramter_vector(P,discoff_len)

    errors = np.array([])
    for i in range(len(w)):
        rvec = w[i][:3, :3]
        tvec = w[i][:3, 3]
        real_coor_temp = np.append(real_coor[i], np.zeros([real_coor[i].shape[0], 1]), 1)
        imagePoints, jacobian = cv2.projectPoints(real_coor_temp, rvec, tvec, A, discoff)
        imagePoints = imagePoints.reshape([-1, 2])
        error = imagePoints - pic_coor[i]
        errors = np.append(errors,error)
    return errors.flatten()



def compose_paramter_vector(A, k):
    alpha = np.array([A[0, 0], A[1, 1], A[0, 2], A[1, 2]])
    alpha = np.append(alpha,k)
    P = alpha
    return P

def decompose_paramter_vector(P,lengthdiscoff):
    [alpha, gamma, uc, vc] = P[0:4]
    discoff = P[4:4+lengthdiscoff]
    A = np.array([[alpha, 0, uc],
                  [0, gamma, vc],
                  [0, 0, 1]])


    return A, discoff

def refine_intrinsic_depth(objPoints_list,imgPoints_list,depth_list,intrinsic,discoff):
    alpha = np.array([intrinsic[0, 0], intrinsic[1, 1],intrinsic[0,2],intrinsic[1,2]])
    lengthdiscoff = np.size(discoff)
    init = np.append(alpha, discoff)
    solver = opt.root(distance_error,init,args=(lengthdiscoff,objPoints_list,imgPoints_list,depth_list),
                      method="lm")
    X = solver.x
    error = distance_error(X,lengthdiscoff,objPoints_list,imgPoints_list,depth_list)
    [alpha, gamma,uc,vc] = X[0:4]
    discoff = np.array([X[4:4 + lengthdiscoff]])
    A = np.array([[alpha, 0, uc],
                  [0, gamma, vc],
                  [0, 0, 1]])
    return A,discoff


def distance_error(X,lengthdiscoff,objPoints_list,imgPoints_list,depth_list):
    error =np.array([])
    [alpha, gamma,uc,vc] = X[0:4]
    discoff = X[4:4 + lengthdiscoff]
    A = np.array([[alpha, 0, uc],
                  [0, gamma, vc],
                  [0, 0, 1]])
    n = len(objPoints_list)
    for i in range(n):
        Point_cam_cood = cv2.undistortPoints(imgPoints_list[i], A, discoff)
        Point_cam_cood = Point_cam_cood.reshape(-1, 2)
        depth_point_acc = np.append(Point_cam_cood, depth_list[i], 1)
        if Point_cam_cood.shape[0]<5:
            continue
        for j in range(depth_point_acc.shape[0]):
            depth_point_acc[j, 0] = Point_cam_cood[j, 0] * depth_point_acc[j, 2]
            depth_point_acc[j, 1] = Point_cam_cood[j,1]*depth_point_acc[j, 2]



        plane = plane_ransac.get_nice_plane(depth_point_acc)
        for j in range(depth_point_acc.shape[0]):
            depth_point_acc[j, 2] = plane[0] * depth_point_acc[j, 0] + plane[1] * depth_point_acc[j, 1] + plane[2]
            # depth_point_acc[j, 0] = Point_cam_cood[j,0]*depth_point_acc[j, 2]
            # depth_point_acc[j, 1] = Point_cam_cood[j,1]*depth_point_acc[j, 2]
        m = Point_cam_cood.shape[0]
        for a in range(int(m/2-1)):
            distance1 = np.linalg.norm(objPoints_list[i][a, :] - objPoints_list[i][m-a-1, :])
            distance2 = np.linalg.norm(depth_point_acc[a, :] - depth_point_acc[m-a-1, :])
            error = np.append(error, np.abs(distance2 - distance1))

    return error

