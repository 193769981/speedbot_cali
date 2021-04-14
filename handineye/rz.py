# -*- coding:utf-8 -*-
import numpy as np
from utils import tranformUtils as utils
from scipy import optimize as op


def loss_function(X, poseList, extrinsicList, board, shot_coor, tags_list_acc):
    rotationX = utils.q2dcm(X[0:4].reshape(1, -1))
    Hx = np.append(rotationX, np.transpose([X[4:7]]), 1)
    Hx = np.append(Hx, np.array([[0, 0, 0, 1]]), 0)
    rotationY = utils.q2dcm(X[7:11].reshape(1, -1))
    Hy = np.append(rotationY, np.transpose([X[11:14]]), 1)
    Hy = np.append(Hy, np.array([[0, 0, 0, 1]]), 0)
    n = len(poseList)

    # error = RMSE(Hx, Hy, poseList, extrinsicList, real_coor)
    error = RMSEcam(Hx, Hy, poseList, board, shot_coor, tags_list_acc)
    return error


def RMSEcam(Hx, Hy, poseList, board, shot_coor, tags_list_acc):
    error = np.array([])
    proj2cam_list = []
    for i in range(len(shot_coor)):
        real_coor = []
        for tag in tags_list_acc[i]:
            center, corner = board.getPointsbyTagId(tag.tag_id)
            real_coor.extend(corner)
        real_coor = np.asarray(real_coor)
        n = np.size(real_coor, 0)
        real_coor = np.append(real_coor.T, np.zeros([1, n]), 0)
        real_coor = np.append(real_coor, np.ones([1, n]), 0)
        # 标定板中角点的个数
        a = np.size(shot_coor[i], 0)
        # 转置，最后加一行1,变成齐次坐标
        shot_coor_i = np.append(shot_coor[i].T, np.ones([1, a]), 0)

        Hbh = np.linalg.inv(np.array(poseList[i]))
        # HT1是板→end的转换，Hbh（base→end）×Hy（板→base）
        HT1 = np.dot(Hbh, Hy)
        # HT2是板→相机，Hx-1（end→相机），HT1（板→end）
        HT2 = np.dot(np.linalg.inv(Hx), HT1)
        # 外参-1（相机→板），HT2（板→相机）
        proj2cam = np.dot(HT2, real_coor)
        proj2cam[:, :] = proj2cam[:, :] / proj2cam[3, :]
        proj2cam_list.append(proj2cam)
        if proj2cam.shape == shot_coor_i.shape:
            error = np.append(error, proj2cam[0:3, :] - shot_coor_i[0:3, :])
        else:
            error = np.append(error, np.array([[np.inf], [np.inf], [np.inf]]))
    return error


def RMSE2cam(Hx, Hy, poseList, board, shot_coor, tags_list_acc):
    '''
    :param Hx: 相机->末端 的转换
    :param Hy: 板->底座 的转换
    :param poseList:
    :param board:
    :param shot_coor:
    :param tags_list_acc:
    :return:
    '''
    error = np.array([])
    proj2cam_list = []
    for i in range(len(shot_coor)):
        real_coor = []
        for tag in tags_list_acc[i]:
            center, corner = board.getPointsbyTagId(tag.tag_id)
            real_coor.extend(corner)
        real_coor = np.asarray(real_coor)
        n = np.size(real_coor, 0)
        real_coor = np.append(real_coor.T, np.zeros([1, n]), 0)
        real_coor = np.append(real_coor, np.ones([1, n]), 0)
        # 标定板中角点的个数
        a = np.size(shot_coor[i], 0)
        # 转置，最后加一行1,变成齐次坐标
        shot_coor_i = np.append(shot_coor[i].T, np.ones([1, a]), 0)

        Hbh = np.linalg.inv(np.array(poseList[i]))
        # HT1是 板→末端 的转换，Hbh（底座→末端）×Hy（板→底座）
        HT1 = np.dot(Hbh, Hy)
        # HT2是板→相机，Hx-1（末端→相机），HT1（板→末端）
        HT2 = np.dot(np.linalg.inv(Hx), HT1)
        # 外参-1（相机→板），HT2（板→相机）
        proj2cam = np.dot(HT2, real_coor)
        proj2cam[:, :] = proj2cam[:, :] / proj2cam[3, :]
        # 从这里开始！
        # proj2cam = proj2cam + np.array([[-9.86806504e-04, -7.50743982e-05, -4.63954602e-04, 1] for i in range(proj2cam.shape[1])]).T
        proj2cam_list.append(proj2cam)

        if proj2cam.shape == shot_coor_i.shape:
            error = np.append(error, np.mean(np.linalg.norm(proj2cam[0:3, :] - shot_coor_i[0:3, :], axis=0)))
        else:
            error = np.append(error, np.mean(np.linalg.norm(np.array([[np.inf], [np.inf], [np.inf]]), axis=0)))
    return error, proj2cam_list


def RMSE(Hx, Hy, poseList, extrinsicList, real_coor, tags_list_acc):
    n = len(poseList)
    error = np.array([])
    # 标定板中角点的个数
    a = np.size(real_coor, 0)
    # 转置，最后加一行0
    real_coor = np.append(real_coor.T, np.zeros([1, a]), 0)
    # 加一行1
    real_coor = np.append(real_coor, np.ones([1, a]), 0)
    for i in range(n):
        Hbh = np.array(poseList[i])
        # HT1是板→底的转换，Hbh（末端→底）×Hy（板→末端）
        HT1 = np.dot(Hbh, Hy)
        # HT2是板→相机，Hx-1（底→相机），HT1（板→底）
        HT2 = np.dot(np.linalg.inv(Hx), HT1)
        # 外参-1（相机→板），HT2（板→相机）
        HT3 = np.dot(np.linalg.inv(extrinsicList[i]), HT2)
        proj = np.dot(HT3, real_coor)

        proj[:, :] = proj[:, :] / proj[3, :]
        error = np.append(error, proj[0:3, :] - real_coor[0:3, :])
    return error


def RMSE2(Hx, Hy, poseList, extrinsicList, real_coor, tags_list_acc):
    n = len(poseList)
    proj2marker_list = []
    proj2cam_list = []
    error = np.array([])
    a = np.size(real_coor, 0)
    real_coor = np.append(real_coor.T, np.zeros([1, a]), 0)
    real_coor = np.append(real_coor, np.ones([1, a]), 0)
    for i in range(n):
        Hbh = np.array(poseList[i])
        HT1 = np.dot(Hbh, Hy)
        HT2 = np.dot(np.linalg.inv(Hx), HT1)
        HT3 = np.dot(np.linalg.inv(extrinsicList[i]), HT2)
        # 计算投影到相机的点
        proj2cam = np.dot(HT2, real_coor)
        proj2cam[:, :] = proj2cam[:, :] / proj2cam[3, :]
        proj2cam_list.append(proj2cam)
        # 计算投影到板上的点
        proj2marker = np.dot(HT3, real_coor)
        proj2marker[:, :] = proj2marker[:, :] / proj2marker[3, :]
        proj2marker_list.append(proj2marker)
        error = np.append(error, np.mean(np.linalg.norm(proj2marker[0:3, :] - real_coor[0:3, :], axis=0)))
    return error, proj2marker_list, proj2cam_list

def refine(X, Y, poseList, extrinsicList, board, shot_coor, tags_list_acc):
    qx = utils.dcm2q(X[:3, :3])
    qy = utils.dcm2q(Y[:3, :3])
    initx = np.append(qx, X[:3, 3])
    inity = np.append(qy, Y[:3, 3])
    init = np.append(initx, inity)
    # solver = op.root(loss_function, init, args=(poseList, extrinsicList, real_coor), method="lm")
    solver = op.root(loss_function, init, args=(poseList, extrinsicList, board, shot_coor, tags_list_acc), method="lm")
    X = solver.x
    rotationX = utils.q2dcm(X[0: 4].reshape(1, -1))
    Hx = np.append(rotationX, np.transpose([X[4:7]]), 1)
    Hx = np.append(Hx, np.array([[0, 0, 0, 1]]), 0)
    rotationY = utils.q2dcm(X[7: 11].reshape(1, -1))
    Hy = np.append(rotationY, np.transpose([X[11:14]]), 1)
    Hy = np.append(Hy, np.array([[0, 0, 0, 1]]), 0)
    return Hx, Hy