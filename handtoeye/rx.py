# -*- coding:utf-8 -*-
import numpy as np
import utils
import math
from utils import tranformUtils as utils
from scipy import optimize as op

def project_error(X,poseList, extrinsicList, real_coor):
    rotation = utils.eulerAngles2RotationMatrix(X[0:3])
    Trc = np.transpose([X[3:6]])
    H = np.append(rotation, Trc, 1)
    H = np.append(H, np.array([[0, 0, 0, 1]]), 0)
    return RMSE(H, poseList, extrinsicList, real_coor)

def RMSE(H, poseList, extrinsicList, real_coor):
    n = len(extrinsicList)
    error = np.array([])
    a = np.size(real_coor, 0)
    real_coor = np.append(real_coor.T, np.zeros([1, a]), 0)
    real_coor = np.append(real_coor, np.ones([1, a]), 0)
    proj_list = []
    for j in range(n - 1):
        # Hg2e是 板→末端: 底→末端 × 板→底(相机→底×板→相机)
        Hg2e = np.dot(np.linalg.inv(poseList[j]), np.dot(H, extrinsicList[j]))
        print("Hg2e是：")
        print(Hg2e)
        print("--------------")
        # HT1是 相机→末端: 底→末端×相机→底
        HT1 = np.dot(np.linalg.inv(poseList[j+1]), H)
        # HT2是 相机→板
        HT2 = np.dot(np.linalg.inv(Hg2e), HT1)
        # HT3是 板→板
        HT3 = np.dot(HT2, extrinsicList[j+1])
        proj = np.dot(HT3, real_coor)
        proj[:, :] = proj[:, :] / proj[3, :]
        proj_list.append(proj)
        error = np.append(error, np.mean(np.linalg.norm(proj[0:3, :] - real_coor[0:3, :], axis=0)))
        # error = np.append(error, abs(proj[:3, :]-real_coor[:3, :]))

    return error

def get_prog(H, poseList, extrinsicList, real_coor):
    n = len(extrinsicList)
    error = np.array([])
    a = np.size(real_coor, 0)
    real_coor = np.append(real_coor.T, np.zeros([1, a]), 0)
    real_coor = np.append(real_coor, np.ones([1, a]), 0)
    proj_list = []
    for j in range(n - 1):
        # Hg2e是 板→末端: 底→末端 × 板→底(相机→底×板→相机)
        Hg2e = np.dot(np.linalg.inv(poseList[j]), np.dot(H, extrinsicList[j]))
        # HT1是 相机→末端: 底→末端×相机→底
        HT1 = np.dot(np.linalg.inv(poseList[j+1]), H)
        # HT2是 相机→板
        HT2 = np.dot(np.linalg.inv(Hg2e), HT1)
        # HT3是 板→板
        HT3 = np.dot(HT2, extrinsicList[j+1])
        proj = np.dot(HT3, real_coor)
        proj[:, :] = proj[:, :] / proj[3, :]
        proj_list.append(proj)

    return proj_list

def refine(init,poseList, extrinsicList, real_coor):
    init_euler = utils.rotationMatrix2EulerAngles(init[:3, :3])
    init = np.append(init_euler, init[:3, 3])
    solver = op.root(project_error, init, args=(poseList, extrinsicList, real_coor), method='lm')
    R = utils.eulerAngles2RotationMatrix(solver.x[:3])
    T = solver.x[3:6]
    H = np.append(R, np.array([T]).T, 1)
    H = np.append(H, np.array([[0, 0, 0, 1]]), 0)
    return H
