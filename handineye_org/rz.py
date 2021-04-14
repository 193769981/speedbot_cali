# -*- coding:utf-8 -*-
import numpy as np
from scipy import optimize as op
import transforms3d

def proj_error(Hx,Hy,poseList, extrinsicList, objpoints):
    #print("proj error")
    #assert "out"
    n = len(poseList)
    error = np.array([])
    a = np.size(objpoints, 0)
    real_coor = np.append(objpoints.T, np.zeros([1, a]), 0)
    real_coor = np.append(real_coor, np.ones([1, a]), 0)
    #print("......................real_coor.....................")
    #print(real_coor)
    for i in range(n):
        Wordpoints = real_coor
        Hbh = np.array(poseList[i])
        proj = np.dot(np.linalg.inv(extrinsicList[i]), np.dot(np.linalg.inv(Hx), np.dot(np.dot(np.linalg.inv(Hbh), Hy), Wordpoints)))
        proj[:, :] = proj[:, :] / proj[3, :]
        error = np.append(error, proj[0:3, :] - real_coor[:3,:])
    return error
def proj_error2(Hx,Hy,poseList, extrinsicList, objpoints):
    #print("proj error")
    #assert "out"
    n = len(poseList)
    error = np.array([])
    a = np.size(objpoints, 0)
    real_coor = np.append(objpoints.T, np.zeros([1, a]), 0)
    real_coor = np.append(real_coor, np.ones([1, a]), 0)
    #print("......................real_coor.....................")
    #print(real_coor)
    for i in range(n):
        Wordpoints = real_coor
        Hbh = np.array(poseList[i])
        proj = np.dot(np.linalg.inv(extrinsicList[i]), np.dot(np.linalg.inv(Hx), np.dot(np.dot(np.linalg.inv(Hbh), Hy), Wordpoints)))
        proj[:, :] = proj[:, :] / proj[3, :]
        error = np.append(error, np.mean(np.abs(proj[0:3, :] - real_coor[:3,:])))
    return error
def proj_error1(Hx,Hy,poseList,depthobjpoints, objpoints):
    n = len(poseList)
    error = np.array([])
    a = np.size(objpoints, 0)
    real_coor = np.append(objpoints.T, np.zeros([1, a]), 0)
    real_coor = np.append(real_coor, np.ones([1, a]), 0)
    real_coor_1=np.append(depthobjpoints.T, np.zeros([1, a]), 0)
    real_coor_1 = np.append(real_coor_1, np.ones([1, a]), 0)
    for i in range(n):
        Wordpoints = real_coor
        Hbh = np.array(poseList[i])
        proj = np.dot(np.dot(np.linalg.inv(Hx), np.dot(np.dot(np.linalg.inv(Hbh), Hy), Wordpoints)))
        proj[:, :] = proj[:, :] / proj[3, :]
        error = np.append(error, proj[0:3, :] - real_coor_1[:3,:])

    return error

def proj_error_each_point(Hx,Hy,poseList, extrinsicList, objpoints):
    n = len(poseList)
    error = np.array([])
    a = np.size(objpoints, 0)
    real_coor = np.append(objpoints.T, np.zeros([1, a]), 0)
    real_coor = np.append(real_coor, np.ones([1, a]), 0)
    for i in range(n):
        Wordpoints = real_coor
        Hbh = np.array(poseList[i])
        proj = np.dot(np.linalg.inv(extrinsicList[i]), np.dot(np.linalg.inv(Hx), np.dot(np.dot(np.linalg.inv(Hbh), Hy), Wordpoints)))
        proj[:, :] = proj[:, :] / proj[3, :]
        error = np.append(error, np.mean(np.abs(proj[0:3, :] - real_coor[:3,:])))

    return error
def loss_function(X,poseList, extrinsicList, real_coor):
    rotationX = transforms3d.quaternions.quat2mat(X[:4])
    Hx = np.append(rotationX, np.transpose([X[4:7]]), 1)
    Hx = np.append(Hx, np.array([[0, 0, 0, 1]]), 0)
    rotationY = transforms3d.quaternions.quat2mat(X[7:11])
    Hy = np.append(rotationY, np.transpose([X[11:14]]), 1)
    Hy = np.append(Hy, np.array([[0, 0, 0, 1]]), 0)
    error =  proj_error(Hx,Hy,poseList, extrinsicList, real_coor)
    return error

def refine(X,Y,poseList, extrinsicList, real_coor):
    qx = transforms3d.quaternions.mat2quat(X[:3, :3])
    qy = transforms3d.quaternions.mat2quat(Y[:3, :3])
    initx = np.append(qx, X[:3, 3])
    inity = np.append(qy, Y[:3, 3])
    init = np.append(initx, inity)
    solver = op.root(loss_function, init, args=(poseList, extrinsicList, real_coor), method="lm")
    X = solver.x
    rotationX = transforms3d.quaternions.quat2mat(X[0:4])
    Hx = np.append(rotationX, np.transpose([X[4:7]]), 1)
    Hx = np.append(Hx, np.array([[0, 0, 0, 1]]), 0)
    rotationY = transforms3d.quaternions.quat2mat(X[7:11])
    Hy = np.append(rotationY, np.transpose([X[11:14]]), 1)
    Hy = np.append(Hy, np.array([[0, 0, 0, 1]]), 0)
    error = proj_error(Hx,Hy,poseList,extrinsicList,real_coor)
    return Hx, Hy