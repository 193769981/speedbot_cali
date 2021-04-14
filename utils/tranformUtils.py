# -*- coding:utf-8 -*-
import numpy as np
import math
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
import transforms3d


def eulerAngles2RotationMatrix(theta):
    """
    将欧拉角转化为旋转矩阵
    :param theta: 类型array 1*3 向量 为欧拉角 单位：角度
    :return: R,类型 ndarray 3*3 旋转矩阵
    """

    #将欧拉角转化为弧度制
    theta = (math.pi / 180) * theta
    R_1 = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])
    R_2 = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])
    R_3 = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_3,np.dot(R_2,R_1))
    return R

def rotationMatrix2EulerAngles(R):
    """
        将旋转矩阵转化为欧拉角
    :param R: 类型 ndarray 3*3 旋转矩阵
    :return: theta 欧拉角，单位角度
    """
    # 将旋转矩阵先转化为旋转向量
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    theta = np.array([])
    if not singular:
        theta = np.append(theta,math.atan2(R[2, 1], R[2, 2]))
        theta = np.append(theta, math.atan2(R[2, 0], sy))
        theta = np.append(theta,  math.atan2(R[1, 0], R[0, 0]))
    else:
        theta = np.append(theta, math.atan2(-R[1, 2], R[1, 1]))
        theta = np.append(theta, math.atan2(-R[2, 0], sy))
        theta = np.append(theta, 0)
    # print('dst:', R)
    theta = (180 / math.pi) * theta
    return theta
def sgn(x):
    if x > 0:
        s = 1
    elif x < 0:
        s = -1
    else:
        s = 0
    s = s + (s == 0)
    return s

def dcm2q(R):
    q = np.array([])
    # if 1 + R[0, 0] - R[1, 1] - R[2, 2] > 0:
    #     q = np.append(q, 0.5 * math.sqrt(1 + R[0, 0] - R[1, 1]  - R[2, 2]) * sgn(R[1, 2]-R[2, 1]))
    # else:
    #     q = np.append(q,0)
    # if 1 + R[0, 0] - R[1, 1] - R[2, 2] > 0:
    #     q = np.append(q, 0.5 * math.sqrt(1 + R[0, 0] - R[1, 1]  - R[2, 2]) * sgn(R[2, 0]-R[0, 2]))
    # else:
    #     q = np.append(q, 0)
    # if 1 - R[0, 0] - R[1, 1] + R[2, 2] > 0:
    #     q = np.append(q, 0.5 * math.sqrt(1 - R[0, 0] - R[1, 1]  + R[2, 2]) * sgn(R[0, 1]-R[1, 0]))
    # else:
    #     q = np.append(q,0)
    # q = np.append(q, 0.5 * math.sqrt(1 + R[0, 0] + R[1, 1]  + R[2, 2]))
    if 1 + R[0, 0] + R[1, 1] + R[2, 2] > 0:
        q0 = 0.5 * math.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
        q = np.append(q, q0)
        q = np.append(q, (R[2, 1] - R[1, 2]) / (4 * q0))
        q = np.append(q, (R[0, 2] - R[2, 0]) / (4 * q0))
        q = np.append(q, (R[1, 0] - R[0, 1]) / (4 * q0))
    else:
        if max(R[0, 0], R[1, 1], R[2, 2]) == R[0, 0]:
            t = math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
            q = np.append(q, (R[2, 1] - R[1, 2]) / t)
            q = np.append(q, t / 4)
            q = np.append(q, (R[0, 2] + R[2, 0]) / t)
            q = np.append(q, (R[0, 1] + R[1, 0]) / t)
        elif max(R[0, 0], R[1, 1], R[2, 2]) == R[1, 1]:
            t = math.sqrt(1 - R[0, 0] + R[1, 1] - R[2, 2])
            q = np.append(q, (R[0, 2] - R[2, 0]) / t)
            q = np.append(q, (R[0, 1] + R[1, 0]) / t)
            q = np.append(q, t / 4)
            q = np.append(q, (R[1, 2] + R[2, 1]) / t)
        else:
            t = math.sqrt(1 - R[0, 0] - R[1, 1] + R[2, 2])
            q = np.append(q, (R[1, 0] - R[0, 1]) / t)
            q = np.append(q, (R[0, 2] + R[2, 0]) / t)
            q = np.append(q, (R[1, 2] - R[2, 1]) / t)
            q = np.append(q, t / 4)
    return q

def q2dcm(q):
    R = np.zeros([3,3],dtype=np.float32)
    q = q/np.linalg.norm(q)
    q0q0 = q[0, 0] * q[0, 0]
    q0q1 = q[0, 0] * q[0, 1]
    q0q2 = q[0, 0] * q[0, 2]
    q0q3 = q[0, 0] * q[0, 3]

    q1q1 = q[0, 1] * q[0, 1]
    q1q2 = q[0, 1] * q[0, 2]
    q1q3 = q[0, 1] * q[0, 3]

    q2q2 = q[0, 2] * q[0, 2]
    q2q3 = q[0, 2] * q[0, 3]

    q3q3 = q[0, 3] * q[0, 3]

    R[0, 0] = 1 - 2 * q2q2 - 2 * q3q3
    R[0, 1] = 2 * (q1q2 - q0q3)
    R[0, 2] = 2 * (q1q3 + q0q2)
    R[1, 0] = 2 * (q1q2 + q0q3)
    R[1, 1] = 1 - 2 * q1q1 - 2 * q3q3
    R[1, 2] = 2 * (q2q3 - q0q1)
    R[2, 0] = 2 * (q1q3 - q0q2)
    R[2, 1] = 2 * (q2q3 + q0q1)
    R[2, 2] = 1 - 2 * q1q1 -2 * q2q2
    # R[0, 0] =  q1q1 - q2q2 - q3q3 + q4q4
    # R[0, 1] = 2 * (q1q2 + q3q4)
    # R[0, 2] = 2 * (q1q3 - q2q4)
    #
    # R[1, 0] = 2 * (q1q2 - q3q4)
    # R[1, 1] = -q1q1 + q2q2 - q3q3 + q4q4
    # R[1, 2] = 2 * (q2q3 + q1q4)
    #
    # R[2, 0] = 2 * (q1q3 + q2q4)
    # R[2, 1] = 2 * (q2q3 - q1q4)
    # R[2, 2] = -q1q1 - q2q2 + q3q3 + q4q4
    return R

def logMatrix(R):
    A = (np.trace(R) - 1) / 2
    if A >1:
        A = 1
    if A < -1:
        A =-1
    fi = math.acos(A)
    if fi != 0:
        w = fi / (2 *math.sin(fi)) * (R - R.T)
    else:
        w = 0.5 * (R - R.T)
    W = np.array([])
    W = np.append(W, w[2, 1])
    W = np.append(W, w[0, 2])
    W = np.append(W, w[1, 0])
    return W.reshape(-1, 1)

def rot2quat(r):
    rv, _ = cv2.Rodrigues(r)
    theta = np.linalg.norm(rv)
    l = rv / theta
    q = np.append(np.sin(theta/2) * l.reshape(-1), [np.cos(theta/2)])
    return q

def quat_mul(q1, q2):
    q3 = np.zeros_like(q1)
    q3[0] = q1[0]*q2[0] - np.dot(q1[1:], q2[1:])
    q3[1:] = np.cross(q1[1:], q2[1:]) + q1[0]*q2[1:] + q2[0]*q1[1:]
    return q3

def rot2dualquat(H):
    q = rot2quat(H[0:3, 0:3])
    qprime = 0.5 * quat_mul(np.concatenate([[0], H[0:3, 3]]), q)
    return q, qprime

def quat2rot(q):
    w,x,y,z = q.tolist()
    r = np.array(
        [
            [1-2*(y*y+z*z), 2*(x*y-z*w),    2*(x*z+y*w)],
            [2*(x*y+z*w),   1-2*(x*x+z*z),  2*(y*z-x*w)],
            [2*(x*z-y*w),   2*(y*z+x*w),    1-2*(x*x+y*y)]
        ]
    )
    return r

def quat_con(q):
    q_inv = -q
    q_inv[0] = - q_inv[0]
    return q_inv

def dualquat2r_t(q1, q1prime):
    r = quat2rot(q1)
    t = (2 * quat_mul(q1prime, quat_con(q1)))[1:]
    t = t.reshape(-1,1)
    return np.append(r,t,1)

def cross_mat(v):
    x,y,z = v.tolist()
    mat = np.array(
        [
            [0,-z,y],
            [z,0,-x],
            [-y,x,0]
        ]
    )
    return mat


def get_sub_mat(qa, qa_prime, qb, qb_prime):
    mat = np.zeros((6,8))
    # equation 1
    mat[:3, 0] = qa[1:] - qb[1:]
    mat[:3, 1:4] = cross_mat(qa[1:] + qb[1:])
    # equation 2
    mat[3:, 0] = qa_prime[1:] - qb_prime[1:]
    mat[3:, 1:4] = cross_mat(qa_prime[1:] + qb_prime[1:])
    mat[3:, 4] = qa[1:] - qb[1:]
    mat[3:, 5:] = cross_mat(qa[1:] + qb[1:])
    return mat

def get_robot_pose(robot_pose, trans_method):
    robot_pose_list = []
    for pose in robot_pose:
        R = transforms3d.euler.euler2mat(float(pose[3]) * math.pi / 180, float(pose[4]) * math.pi / 180,
                                         float(pose[5]) * math.pi / 180, trans_method)
        robot_pose = np.zeros([4, 4])
        robot_pose[:3, :3] = R[:, :]
        robot_pose[0, 3] = float(pose[0]) / 1000
        robot_pose[1, 3] = float(pose[1]) / 1000
        robot_pose[2, 3] = float(pose[2]) / 1000
        robot_pose[3, 3] = 1.0
        robot_pose_list.append(robot_pose)
    return robot_pose_list