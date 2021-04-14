#-*- coding:utf-8 -*-
import cv2
import transforms3d
import numpy as np
from method import tsai
from method import dual
from method import li
import os
from scipy import optimize as opt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import multiprocessing
from utils import load_data

from multiprocessing import Pool
import random
import threading
from numba import jit
import numba as nb
from progressbar import *

# @jit(nopython=True)
def score_std_handineye(expect_camera_list,Hend2base,Hobj2camera,Hx):
    expect_robot_pose = np.zeros((expect_camera_list.shape[0], 4, 4), dtype=np.float32)
    score = np.zeros((expect_camera_list.shape[0],1),dtype=np.float32)
    expect_robot_q0 = np.zeros((Hend2base.shape[0],1),dtype=np.float32)
    expect_robot_q1 = np.zeros((Hend2base.shape[0],1),dtype=np.float32)
    expect_robot_q2 = np.zeros((Hend2base.shape[0],1),dtype=np.float32)
    expect_robot_q3 = np.zeros((Hend2base.shape[0],1),dtype=np.float32)
    expect_robot_t0 = np.zeros((Hend2base.shape[0],1),dtype=np.float32)
    expect_robot_t1 = np.zeros((Hend2base.shape[0],1),dtype=np.float32)
    expect_robot_t2 = np.zeros((Hend2base.shape[0],1),dtype=np.float32)
    for i in range(expect_camera_list.shape[0]):
        for j in range(Hend2base.shape[0]):
            robot_pose= np.dot(Hend2base[j], np.dot(Hx, np.dot(Hobj2camera[j], np.dot(
                    np.linalg.inv(expect_camera_list[i]), np.linalg.inv(Hx)))))
            R = robot_pose[:3,:3]
            Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = R.flat
            K = np.array([
                [Qxx - Qyy - Qzz, 0, 0, 0],
                [Qyx + Qxy, Qyy - Qxx - Qzz, 0, 0],
                [Qzx + Qxz, Qzy + Qyz, Qzz - Qxx - Qyy, 0],
                [Qyz - Qzy, Qzx - Qxz, Qxy - Qyx, Qxx + Qyy + Qzz]],dtype=np.float32
            ) / 3.0
            vals, vecs = np.linalg.eigh(K)
            q = vecs[:,np.argmax(vals)]
            if j==0:
                expect_robot_q0[j,0]=q[3]
                expect_robot_q1[j,0]=q[0]
                expect_robot_q2[j,0]=q[1]
                expect_robot_q3[j,0]=q[2]
            else:
                sub = abs(q[3]-expect_robot_q0[0,0])+abs(q[0]-expect_robot_q1[0,0])+abs(q[1]-expect_robot_q2[0,0])+abs(q[2]-expect_robot_q3[0,0])
                sum = abs(q[3]+expect_robot_q0[0,0])+abs(q[0]+expect_robot_q1[0,0])+abs(q[1]+expect_robot_q2[0,0])+abs(q[2]+expect_robot_q3[0,0])
                if sub<sum:
                    expect_robot_q0[j, 0] = q[3]
                    expect_robot_q1[j, 0] = q[0]
                    expect_robot_q2[j, 0] = q[1]
                    expect_robot_q3[j, 0] = q[2]
                else:
                    expect_robot_q0[j, 0] = -q[3]
                    expect_robot_q1[j, 0] = -q[0]
                    expect_robot_q2[j, 0] = -q[1]
                    expect_robot_q3[j, 0] = -q[2]

            # if 1 + R[0, 0] + R[1, 1] + R[2, 2] > 0:
            #     q0 = 0.5 * math.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
            #     expect_robot_q0[j, 0] = q0
            #     expect_robot_q1[j, 0] = (R[2, 1] - R[1, 2]) / (4 * q0)
            #     expect_robot_q2[j, 0] = (R[0, 2] - R[2, 0]) / (4 * q0)
            #     expect_robot_q3[j, 0] = (R[1, 0] - R[0, 1]) / (4 * q0)
            # else:
            #     if max(R[0, 0], R[1, 1], R[2, 2]) == R[0, 0]:
            #         t = math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
            #         expect_robot_q0[j, 0] = (R[2, 1] - R[1, 2]) / t
            #         expect_robot_q1[j, 0] = t / 4
            #         expect_robot_q2[j, 0] = (R[0, 2] + R[2, 0]) / t
            #         expect_robot_q3[j, 0] = (R[0, 1] + R[1, 0]) / t
            #     elif max(R[0, 0], R[1, 1], R[2, 2]) == R[1, 1]:
            #         t = math.sqrt(1 - R[0, 0] + R[1, 1] - R[2, 2])
            #         expect_robot_q0[j, 0] = (R[0, 2] - R[2, 0]) / t
            #         expect_robot_q1[j, 0] = (R[0, 1] + R[1, 0]) / t
            #         expect_robot_q2[j, 0] = t / 4
            #         expect_robot_q3[j, 0] = (R[1, 2] + R[2, 1]) / t
            #     else:
            #         t = math.sqrt(1 - R[0, 0] - R[1, 1] + R[2, 2])
            #         expect_robot_q0[j, 0] = (R[1, 0] - R[0, 1]) / t
            #         expect_robot_q1[j, 0] = (R[0, 2] + R[2, 0]) / t
            #         expect_robot_q2[j, 0] = (R[1, 2] - R[2, 1]) / t
            #         expect_robot_q3[j, 0] = t / 4
            expect_robot_t0[j,0]=robot_pose[0,3]
            expect_robot_t1[j,0]=robot_pose[1,3]
            expect_robot_t2[j,0]=robot_pose[2,3]
        # print("p0",expect_robot_q0)
        # print("p1",expect_robot_q1)
        # print("p2",expect_robot_q2)
        # print("p3",expect_robot_q3)
        expect_robot_q0_std = np.std(expect_robot_q0)
        expect_robot_q1_std = np.std(expect_robot_q1)
        expect_robot_q2_std = np.std(expect_robot_q2)
        expect_robot_q3_std = np.std(expect_robot_q3)
        expect_robot_t0_std = np.std(expect_robot_t0)
        expect_robot_t1_std = np.std(expect_robot_t1)
        expect_robot_t2_std = np.std(expect_robot_t2)
        expect_robot_q0_mean = np.mean(expect_robot_q0)
        expect_robot_q1_mean = np.mean(expect_robot_q1)
        expect_robot_q2_mean = np.mean(expect_robot_q2)
        expect_robot_q3_mean = np.mean(expect_robot_q3)
        expect_robot_t0_mean = np.mean(expect_robot_t0)
        expect_robot_t1_mean = np.mean(expect_robot_t1)
        expect_robot_t2_mean = np.mean(expect_robot_t2)

        score[i,0] = expect_robot_q0_std+expect_robot_q1_std+expect_robot_q2_std+\
                expect_robot_q3_std+expect_robot_t0_std+expect_robot_t1_std+expect_robot_t1_std+expect_robot_t2_std
        w = expect_robot_q0_mean
        x = expect_robot_q1_mean
        y = expect_robot_q2_mean
        z = expect_robot_q3_mean
        Nq = w * w + x * x + y * y + z * z
        R = np.zeros((3,3),dtype=np.float32)
        if Nq < 10^-6:
            R =  np.eye(3,dtype=np.float32)
        else:
            s = 2.0 / Nq
            X = x * s
            Y = y * s
            Z = z * s
            wX = w * X
            wY = w * Y
            wZ = w * Z
            xX = x * X
            xY = x * Y
            xZ = x * Z
            yY = y * Y
            yZ = y * Z
            zZ = z * Z
            R = np.array(
               [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
                [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
                [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]],dtype=np.float32)

        expect_robot_pose[i,:3,:3] = R[:,:]
        expect_robot_pose[i,0,3] = expect_robot_t0_mean
        expect_robot_pose[i,1,3] = expect_robot_t1_mean
        expect_robot_pose[i,2,3] = expect_robot_t2_mean
        expect_robot_pose[i,3,3] = 1
    return score,expect_robot_pose

# @jit(nopython=True)
def score_std_handoneye(expect_camera_list, Hend2base, Hobj2camera, Hx):
    expect_robot_pose = np.zeros((expect_camera_list.shape[0], 4, 4), dtype=np.float32)
    score = np.zeros((expect_camera_list.shape[0], 1), dtype=np.float32)
    expect_robot_q0 = np.zeros((Hend2base.shape[0], 1), dtype=np.float32)
    expect_robot_q1 = np.zeros((Hend2base.shape[0], 1), dtype=np.float32)
    expect_robot_q2 = np.zeros((Hend2base.shape[0], 1), dtype=np.float32)
    expect_robot_q3 = np.zeros((Hend2base.shape[0], 1), dtype=np.float32)
    expect_robot_t0 = np.zeros((Hend2base.shape[0], 1), dtype=np.float32)
    expect_robot_t1 = np.zeros((Hend2base.shape[0], 1), dtype=np.float32)
    expect_robot_t2 = np.zeros((Hend2base.shape[0], 1), dtype=np.float32)
    for i in range(expect_camera_list.shape[0]):
        for j in range(Hend2base.shape[0]):
            robot_pose = np.dot(Hx,np.dot(expect_camera_list[i], np.dot(np.linalg.inv(Hobj2camera[j]),
                                                                            np.dot(np.linalg.inv(Hx), Hend2base[j]))))
            R = robot_pose[:3, :3]
            Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = R.flat
            K = np.array([
                [Qxx - Qyy - Qzz, 0, 0, 0],
                [Qyx + Qxy, Qyy - Qxx - Qzz, 0, 0],
                [Qzx + Qxz, Qzy + Qyz, Qzz - Qxx - Qyy, 0],
                [Qyz - Qzy, Qzx - Qxz, Qxy - Qyx, Qxx + Qyy + Qzz]], dtype=np.float32
            ) / 3.0
            vals, vecs = np.linalg.eigh(K)
            q = vecs[:, np.argmax(vals)]
            if j == 0:
                expect_robot_q0[j, 0] = q[3]
                expect_robot_q1[j, 0] = q[0]
                expect_robot_q2[j, 0] = q[1]
                expect_robot_q3[j, 0] = q[2]
            else:
                sub = abs(q[3] - expect_robot_q0[0, 0]) + abs(q[0] - expect_robot_q1[0, 0]) + abs(
                    q[1] - expect_robot_q2[0, 0]) + abs(q[2] - expect_robot_q3[0, 0])
                sum = abs(q[3] + expect_robot_q0[0, 0]) + abs(q[0] + expect_robot_q1[0, 0]) + abs(
                    q[1] + expect_robot_q2[0, 0]) + abs(q[2] + expect_robot_q3[0, 0])
                if sub < sum:
                    expect_robot_q0[j, 0] = q[3]
                    expect_robot_q1[j, 0] = q[0]
                    expect_robot_q2[j, 0] = q[1]
                    expect_robot_q3[j, 0] = q[2]
                else:
                    expect_robot_q0[j, 0] = -q[3]
                    expect_robot_q1[j, 0] = -q[0]
                    expect_robot_q2[j, 0] = -q[1]
                    expect_robot_q3[j, 0] = -q[2]
            expect_robot_t0[j, 0] = robot_pose[0, 3]
            expect_robot_t1[j, 0] = robot_pose[1, 3]
            expect_robot_t2[j, 0] = robot_pose[2, 3]
        expect_robot_q0_std = np.std(expect_robot_q0)
        expect_robot_q1_std = np.std(expect_robot_q1)
        expect_robot_q2_std = np.std(expect_robot_q2)
        expect_robot_q3_std = np.std(expect_robot_q3)
        expect_robot_t0_std = np.std(expect_robot_t0)
        expect_robot_t1_std = np.std(expect_robot_t1)
        expect_robot_t2_std = np.std(expect_robot_t2)
        expect_robot_q0_mean = np.mean(expect_robot_q0)
        expect_robot_q1_mean = np.mean(expect_robot_q1)
        expect_robot_q2_mean = np.mean(expect_robot_q2)
        expect_robot_q3_mean = np.mean(expect_robot_q3)
        expect_robot_t0_mean = np.mean(expect_robot_t0)
        expect_robot_t1_mean = np.mean(expect_robot_t1)
        expect_robot_t2_mean = np.mean(expect_robot_t2)

        score[i, 0] = expect_robot_q0_std + expect_robot_q1_std + expect_robot_q2_std + \
                      expect_robot_q3_std + expect_robot_t0_std + expect_robot_t1_std + expect_robot_t1_std + expect_robot_t2_std
        w = expect_robot_q0_mean
        x = expect_robot_q1_mean
        y = expect_robot_q2_mean
        z = expect_robot_q3_mean
        Nq = w * w + x * x + y * y + z * z
        R = np.zeros((3, 3), dtype=np.float32)
        if Nq < 10 ^ -6:
            R = np.eye(3, dtype=np.float32)
        else:
            s = 2.0 / Nq
            X = x * s
            Y = y * s
            Z = z * s
            wX = w * X
            wY = w * Y
            wZ = w * Z
            xX = x * X
            xY = x * Y
            xZ = x * Z
            yY = y * Y
            yZ = y * Z
            zZ = z * Z
            R = np.array(
                [[1.0 - (yY + zZ), xY - wZ, xZ + wY],
                 [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
                 [xZ - wY, yZ + wX, 1.0 - (xX + yY)]], dtype=np.float32)

        expect_robot_pose[i, :3, :3] = R[:, :]
        expect_robot_pose[i, 0, 3] = expect_robot_t0_mean
        expect_robot_pose[i, 1, 3] = expect_robot_t1_mean
        expect_robot_pose[i, 2, 3] = expect_robot_t2_mean
        expect_robot_pose[i, 3, 3] = 1
    return score, expect_robot_pose

# @jit(nopython=True)
def score_no_local(expect_robot_pose, Hend2base):
    score = np.zeros((expect_robot_pose.shape[0], 1), dtype=np.float32)
    for i in range(expect_robot_pose.shape[0]):
        min_score = 0
        R = expect_robot_pose[i, :3, :3]
        Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = R.astype(np.float32).flat
        K = np.zeros((4, 4), dtype=np.float32)
        K[0, 0] = Qxx - Qyy - Qzz
        K[1, 0] = Qyx + Qxy
        K[1, 1] = Qyy - Qxx - Qzz
        K[2, 0] = Qzx + Qxz
        K[2, 1] = Qzy + Qyz
        K[2, 2] = Qzz - Qxx - Qyy
        K[3, 0] = Qyz - Qzy
        K[3, 1] = Qzx - Qxz
        K[3, 2] = Qxy - Qyx
        K[3, 3] = Qxx + Qyy + Qzz
        K = K / 3.0
        vals, vecs = np.linalg.eigh(K)
        q0 = vecs[:, np.argmax(vals)]
        t0 = expect_robot_pose[i, :3, 3]
        score[i, 0] = 0
        for j in range(Hend2base.shape[0]):
            R = Hend2base[j, :3, :3]
            Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = R.flat
            K = np.array([
                [Qxx - Qyy - Qzz, 0, 0, 0],
                [Qyx + Qxy, Qyy - Qxx - Qzz, 0, 0],
                [Qzx + Qxz, Qzy + Qyz, Qzz - Qxx - Qyy, 0],
                [Qyz - Qzy, Qzx - Qxz, Qxy - Qyx, Qxx + Qyy + Qzz]], dtype=np.float32
            ) / 3.0
            vals, vecs = np.linalg.eigh(K)
            q = vecs[:, np.argmax(vals)]
            t = Hend2base[j, :3, 3]
            q_dis = min(np.linalg.norm(q + q0), np.linalg.norm(q - q0))
            t_dis = np.linalg.norm(t - t0)
            if q_dis<0.2 and t_dis<0.22:
                # score_t = -(math.pow(math.e, -0.5 * abs(q_dis)) + 1 * math.pow(math.e, -0.5 * abs(t_dis)))
                # if score_t < min_score:
                #     min_score = score_t
                score[i, 0] = -1

            #
        #score[i, 0] = min_score
    return score

# @jit(nopython=True)
def getRobotPose_handineye(expect_camera_list, Hend2base, Hobj2camera,Hx):
    expect_robot_pose = np.zeros((expect_camera_list.shape[0],4,4),dtype=np.float32)
    expect_robot_q0 = np.zeros((Hend2base.shape[0],1),dtype=np.float32)
    expect_robot_q1 = np.zeros((Hend2base.shape[0],1),dtype=np.float32)
    expect_robot_q2 = np.zeros((Hend2base.shape[0],1),dtype=np.float32)
    expect_robot_q3 = np.zeros((Hend2base.shape[0],1),dtype=np.float32)
    expect_robot_t0 = np.zeros((Hend2base.shape[0],1),dtype=np.float32)
    expect_robot_t1 = np.zeros((Hend2base.shape[0],1),dtype=np.float32)
    expect_robot_t2 = np.zeros((Hend2base.shape[0],1),dtype=np.float32)
    for i in range(expect_camera_list.shape[0]):
        for j in range(Hend2base.shape[0]):
            robot_pose= np.dot(Hend2base[j], np.dot(Hx, np.dot(Hobj2camera[j], np.dot(
                    np.linalg.inv(expect_camera_list[i]), np.linalg.inv(Hx)))))
            R = robot_pose[:3,:3]
            Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = R.flat
            K = np.array([
                [Qxx - Qyy - Qzz, 0, 0, 0],
                [Qyx + Qxy, Qyy - Qxx - Qzz, 0, 0],
                [Qzx + Qxz, Qzy + Qyz, Qzz - Qxx - Qyy, 0],
                [Qyz - Qzy, Qzx - Qxz, Qxy - Qyx, Qxx + Qyy + Qzz]],dtype=np.float32
            ) / 3.0
            vals, vecs = np.linalg.eigh(K)
            q = vecs[:,np.argmax(vals)]
            if j==0:
                expect_robot_q0[j,0]=q[3]
                expect_robot_q1[j,0]=q[0]
                expect_robot_q2[j,0]=q[1]
                expect_robot_q3[j,0]=q[2]
            else:
                sub = abs(q[3]-expect_robot_q0[0,0])+abs(q[0]-expect_robot_q1[0,0])+abs(q[1]-expect_robot_q2[0,0])+abs(q[2]-expect_robot_q3[0,0])
                sum = abs(q[3]+expect_robot_q0[0,0])+abs(q[0]+expect_robot_q1[0,0])+abs(q[1]+expect_robot_q2[0,0])+abs(q[2]+expect_robot_q3[0,0])
                if sub<sum:
                    expect_robot_q0[j, 0] = q[3]
                    expect_robot_q1[j, 0] = q[0]
                    expect_robot_q2[j, 0] = q[1]
                    expect_robot_q3[j, 0] = q[2]
                else:
                    expect_robot_q0[j, 0] = -q[3]
                    expect_robot_q1[j, 0] = -q[0]
                    expect_robot_q2[j, 0] = -q[1]
                    expect_robot_q3[j, 0] = -q[2]
            expect_robot_t0[j,0]=robot_pose[0,3]
            expect_robot_t1[j,0]=robot_pose[1,3]
            expect_robot_t2[j,0]=robot_pose[2,3]
        expect_robot_q0_mean = np.mean(expect_robot_q0)
        expect_robot_q1_mean = np.mean(expect_robot_q1)
        expect_robot_q2_mean = np.mean(expect_robot_q2)
        expect_robot_q3_mean = np.mean(expect_robot_q3)
        expect_robot_t0_mean = np.mean(expect_robot_t0)
        expect_robot_t1_mean = np.mean(expect_robot_t1)
        expect_robot_t2_mean = np.mean(expect_robot_t2)
        w = expect_robot_q0_mean
        x = expect_robot_q1_mean
        y = expect_robot_q2_mean
        z = expect_robot_q3_mean
        Nq = w * w + x * x + y * y + z * z
        R = np.zeros((3,3),dtype=np.float32)
        if Nq < 10^-6:
            R =  np.eye(3,dtype=np.float32)
        else:
            s = 2.0 / Nq
            X = x * s
            Y = y * s
            Z = z * s
            wX = w * X
            wY = w * Y
            wZ = w * Z
            xX = x * X
            xY = x * Y
            xZ = x * Z
            yY = y * Y
            yZ = y * Z
            zZ = z * Z
            R = np.array(
               [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
                [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
                [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]],dtype=np.float32)

        expect_robot_pose[i,:3,:3] = R[:,:]
        expect_robot_pose[i,0,3] = expect_robot_t0_mean
        expect_robot_pose[i,1,3] = expect_robot_t1_mean
        expect_robot_pose[i,2,3] = expect_robot_t2_mean
        expect_robot_pose[i,3,3] = 1
    return expect_robot_pose

# @jit(nopython=True)
def getRobotPose_handoneye(expect_camera_list, Hend2base, Hobj2camera,Hx):
    expect_robot_pose = np.zeros((expect_camera_list.shape[0],4,4),dtype=np.float32)
    expect_robot_q0 = np.zeros((Hend2base.shape[0],1),dtype=np.float32)
    expect_robot_q1 = np.zeros((Hend2base.shape[0],1),dtype=np.float32)
    expect_robot_q2 = np.zeros((Hend2base.shape[0],1),dtype=np.float32)
    expect_robot_q3 = np.zeros((Hend2base.shape[0],1),dtype=np.float32)
    expect_robot_t0 = np.zeros((Hend2base.shape[0],1),dtype=np.float32)
    expect_robot_t1 = np.zeros((Hend2base.shape[0],1),dtype=np.float32)
    expect_robot_t2 = np.zeros((Hend2base.shape[0],1),dtype=np.float32)
    for i in range(expect_camera_list.shape[0]):
        for j in range(Hend2base.shape[0]):
            robot_pose = np.dot(np.dot(Hx, expect_camera_list[i]), np.dot(np.linalg.inv(Hobj2camera[j]), np.dot(np.linalg.inv(Hx), Hend2base[j])))
            R = robot_pose[:3,:3]
            Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = R.flat
            K = np.array([
                [Qxx - Qyy - Qzz, 0, 0, 0],
                [Qyx + Qxy, Qyy - Qxx - Qzz, 0, 0],
                [Qzx + Qxz, Qzy + Qyz, Qzz - Qxx - Qyy, 0],
                [Qyz - Qzy, Qzx - Qxz, Qxy - Qyx, Qxx + Qyy + Qzz]],dtype=np.float32
            ) / 3.0
            vals, vecs = np.linalg.eigh(K)
            q = vecs[:,np.argmax(vals)]
            if j==0:
                expect_robot_q0[j,0]=q[3]
                expect_robot_q1[j,0]=q[0]
                expect_robot_q2[j,0]=q[1]
                expect_robot_q3[j,0]=q[2]
            else:
                sub = abs(q[3]-expect_robot_q0[0,0])+abs(q[0]-expect_robot_q1[0,0])+abs(q[1]-expect_robot_q2[0,0])+abs(q[2]-expect_robot_q3[0,0])
                sum = abs(q[3]+expect_robot_q0[0,0])+abs(q[0]+expect_robot_q1[0,0])+abs(q[1]+expect_robot_q2[0,0])+abs(q[2]+expect_robot_q3[0,0])
                if sub<sum:
                    expect_robot_q0[j, 0] = q[3]
                    expect_robot_q1[j, 0] = q[0]
                    expect_robot_q2[j, 0] = q[1]
                    expect_robot_q3[j, 0] = q[2]
                else:
                    expect_robot_q0[j, 0] = -q[3]
                    expect_robot_q1[j, 0] = -q[0]
                    expect_robot_q2[j, 0] = -q[1]
                    expect_robot_q3[j, 0] = -q[2]
            expect_robot_t0[j,0]=robot_pose[0,3]
            expect_robot_t1[j,0]=robot_pose[1,3]
            expect_robot_t2[j,0]=robot_pose[2,3]
        expect_robot_q0_mean = np.mean(expect_robot_q0)
        expect_robot_q1_mean = np.mean(expect_robot_q1)
        expect_robot_q2_mean = np.mean(expect_robot_q2)
        expect_robot_q3_mean = np.mean(expect_robot_q3)
        expect_robot_t0_mean = np.mean(expect_robot_t0)
        expect_robot_t1_mean = np.mean(expect_robot_t1)
        expect_robot_t2_mean = np.mean(expect_robot_t2)
        w = expect_robot_q0_mean
        x = expect_robot_q1_mean
        y = expect_robot_q2_mean
        z = expect_robot_q3_mean
        Nq = w * w + x * x + y * y + z * z
        R = np.zeros((3,3),dtype=np.float32)
        if Nq < 10^-6:
            R =  np.eye(3,dtype=np.float32)
        else:
            s = 2.0 / Nq
            X = x * s
            Y = y * s
            Z = z * s
            wX = w * X
            wY = w * Y
            wZ = w * Z
            xX = x * X
            xY = x * Y
            xZ = x * Z
            yY = y * Y
            yZ = y * Z
            zZ = z * Z
            R = np.array(
               [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
                [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
                [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]],dtype=np.float32)

        expect_robot_pose[i,:3,:3] = R[:,:]
        expect_robot_pose[i,0,3] = expect_robot_t0_mean
        expect_robot_pose[i,1,3] = expect_robot_t1_mean
        expect_robot_pose[i,2,3] = expect_robot_t2_mean
        expect_robot_pose[i,3,3] = 1
    return expect_robot_pose

def multi_score(cali_type, Hend2base, Hobj2camera, method, Hx, expect_camera_list):
    '''
    根据得分大小，对相机位姿进行排序
    :param cali_type:
    :param Hend2base:
    :param Hobj2camera:
    :param method:
    :param Hx:
    :param expect_camera_list:
    :return: 排序后的相机位姿（外参）列表
    '''
    expect_cameras = np.array(expect_camera_list)
    Hend2bases = np.array(Hend2base)
    Hobj2cameras = np.array(Hobj2camera)
    if method == 1 or method == 3:
        if cali_type == 0:
            score, robot_pose = score_std_handineye(expect_cameras, Hend2bases, Hobj2cameras, Hx)
        else:
            score, robot_pose = score_std_handoneye(expect_cameras, Hend2bases, Hobj2cameras, Hx)
        if method == 3:
            score2 = score_no_local(robot_pose, Hend2bases)
            score = score * 10 + score2
    elif method == 0:
        if cali_type == 0:
            robot_pose = getRobotPose_handineye(expect_cameras, Hend2bases, Hobj2cameras, Hx)
        else:
            robot_pose = getRobotPose_handoneye(expect_cameras, Hend2bases, Hobj2cameras, Hx)
        score = score_no_local(robot_pose, Hend2bases)
    score_list = score.tolist()
    sort_list = [[a, b] for a, b in zip(score_list, expect_camera_list)]
    # sort_list = map(list,zip(score_list,expect_camera_list))

    # 根据得分大小排序
    sort_list.sort(key=lambda x: x[0])
    campose_order_list = []
    for t in sort_list:
        campose_order_list.append(t[1])
    campose_order_list.reverse()
    return campose_order_list

from AprilTag.aprilTagUtils import *
from visualization import get3dpoints
def getCameraposeFromDepth(board ,rgbimg,depthimg,camintrinsic, camdist):   #bug
    discoff = camdist
    camera_matrix = camintrinsic

    # 获取图片size
    img = rgbimg
    img_size = tuple([list(img.shape)[1], list(img.shape)[0]])

    shot_coor = []
    img_data = rgbimg
    pclz = depthimg
    # 图像去畸变
    img_data = cv2.undistort(img_data, camera_matrix, discoff)
    pclz = cv2.undistort(pclz, camera_matrix, discoff)

    tags = detectTags_img(board, img_data, camera_matrix, verbose=0)
    corners = [tag.corners for tag in tags]
    for corner in corners:
        delete_flag = 0
        # 若marker板没拍摄完全，计算出来的角点会超过图片边界
        for n in range(4):
            if round(corner[n][1]) >= list(img_size)[1]:
                print("width超过图片边界({}),已删除该tag".format(corner[n][1]))
                delete_flag = 1

            if round(corner[n][0]) >= list(img_size)[0]:
                print("height超过图片边界({}),已删除该tag".format(corner[n][0]))
                delete_flag = 1

        # 如果有超界的点，则越过,不记录该tag
        if delete_flag:
            continue

        shot_coor.append([corner[0][0], corner[0][1], pclz[int(round(corner[0][1])), int(round(corner[0][0]))]])
        shot_coor.append([corner[1][0], corner[1][1], pclz[int(round(corner[1][1])), int(round(corner[1][0]))]])
        shot_coor.append([corner[2][0], corner[2][1], pclz[int(round(corner[2][1])), int(round(corner[2][0]))]])
        shot_coor.append([corner[3][0], corner[3][1], pclz[int(round(corner[3][1])), int(round(corner[3][0]))]])
    if shot_coor == []:
        return False,None,None,None
    shot_coor = np.asarray(shot_coor)

    xyz = get3dpoints.depth2xyz(shot_coor, camera_matrix, img_size, flatten=True, disrete=True)
    #real_shot_coor_list.append(xyz)

    marker_coor = []
    for tag in tags:
        _, real_marker_corner = board.getPointsbyTagId(tag.tag_id)
        marker_coor.extend(real_marker_corner)
    marker_coor = np.asarray(marker_coor)
    n = np.size(marker_coor, 0)
    marker_coor = np.append(marker_coor, np.zeros([n, 1]), 1)
    # 计算得到所有数据的外参
    from utils import depthUtils
    if (len(marker_coor) != len(xyz)):
        return False, None, None, None
    extrinsic_list = depthUtils.get_camerapose_by_depth_one(marker_coor, xyz)


    return True,extrinsic_list,xyz,marker_coor


class auto_handeye_calibration(object):
    def __init__(self, board, robot, camera, config, cali_type, minZ_qt , maxZ_qt , angle_qt):
        '''
        初始化，需要指定标定板，机器臂，相机，以及初始化文件
        :param board: 标定板
        :param robot: 机器臂
        :param camera: 相机
        :param config: 配置文件
        '''
        self.config = config
        self.board = board
        self.robot = robot
        self.camera = camera
        self.cali_type = cali_type  # calibration type： handineye 0 handtoeye 1

        fs = cv2.FileStorage(config, cv2.FILE_STORAGE_READ)
        self.minZ = float(minZ_qt)
        self.maxZ = float(maxZ_qt)
        self.inter_z = fs.getNode("inter_z").real()
        self.inter_xy = fs.getNode("inter_xy").real()
        self.optic_angle = float(angle_qt)
        self.picture_number = int(fs.getNode("picture_number").real())
        self.save_txt = []
        fs.release()

        # pose -- matrix
        # init_pose_str -- angle

        #临时注释
        print("connect robot")
        flag, init_pose_str, pose = robot.get_init_pose()

        if flag:
            self.init_robot_pose = pose
            self.init_robot_pose_str = init_pose_str
            self.next_step_method = 3
        else:
            str1 = 'msg|'+'robot connect failed,please contact Robot Engineer'
            camera.sendProcessBar(str1)
            assert False,str1
            self.init_robot_pose = None
        print("get init success")
        print(pose)
    def set_init_robot_pose(self, pose):
        self.init_robot_pose = pose.copy()

    def init_handeye(self):
        '''
        通过旋转机械臂来初始化手眼标定，得到较为粗略的手眼矩阵
        :return:
        '''

        # flag,rgb_image = self.move_lock.move_and_getimage(self.init_robot_pose)
        # 初次获取robot_pose,得到一组数据,计算出来外参
        # angle
        flag, robot_pose_str = self.robot.init_move(self.init_robot_pose_str)
        assert flag, "该位姿机器人不可达!"
        # import pdb
        # pdb.set_trace()
        if self.camera.type=='3d':
            print('3d')
            flag, rgb, depth, img_RootPath = self.camera.get_rgb_depth_image()
        else:
            print('2d')
            flag, rgb, img_RootPath = self.camera.get_rgb_image()
        assert flag, "拍摄失败!"
        # cv2.imshow('rgb',rgb)
        # cv2.waitKey(0)
        #print("noget")
        # rgb = cv2.undistort(rgb, self.camera.intrinsic,self.camera.dist)
        flag, objpoint, imgpoint = self.board.getObjImgPointList(rgb)
        #circle
        # flag, objpoint, imgpoint = self.board.getObjImgPointListFromCircle(rgb,5,4,0)
        # for i in range(objpoint.shape[0]):
        #     print((int(imgpoint[i][0]),int(imgpoint[i][1])))
        #     cv2.circle(rgb,(int(imgpoint[i][0]),int(imgpoint[i][1])),1,(0,255,0),2)
        #     # cv2.nameWindow('rgb',0)
        #     cv2.imshow('rgb',rgb)
        #     print(objpoint[i])
        #     cv2.waitKey(0)
        #print("get")
        print('intr and dist',self.camera.intrinsic,self.camera.dist)
        if not flag:
            str1='msg|该位姿下标定板角点无法被拍摄到,please adjust board location!'
            self.camera.sendProcessBar(str1)
        assert flag, "该位姿下标定板角点无法被拍摄到!"

        if self.camera.type=='3d':
            flag, camerapose, depthcorner, markercorner = getCameraposeFromDepth(self.board, rgb, depth,self.camera.intrinsic, self.camera.dist)
            self.depth_image = []
            self.depth_corner = []  # get corner in camera coordidate from depth and camera intrinsic
            self.marker_corner = []  # get corner in world coordidate from real marker size
            self.depth_image.append(depth)
            self.depth_corner.append(depthcorner)
            self.marker_corner.append(markercorner)
            print('camerapose svd')
            # print(flag)
            print(camerapose)

            camerapose = self.board.extrinsic(imgpoint, objpoint, self.camera.intrinsic, self.camera.dist)
            print('camerapose pnp', camerapose)
        else:
            camerapose = self.board.extrinsic(imgpoint, objpoint, self.camera.intrinsic, self.camera.dist)
        # assert flag, "获取相机外参失败!"
        self.objpoint_list = []
        self.imgpoint_list = []
        self.Hend2base = []
        self.Hobj2camera = []
        self.rgb_image = []
        self.result = []
        self.depth_corner=[]  #get corner in camera coordidate from depth and camera intrinsic
        self.marker_corner=[] #get corner in world coordidate from real marker size
        self.objpoint_list.append(objpoint)
        self.imgpoint_list.append(imgpoint)
        self.rgb_image.append(rgb)
        self.Hend2base.append(self.init_robot_pose)
        self.save_txt.append(robot_pose_str)
        self.Hobj2camera.append(camerapose)
        # self.robot.trans_method = ''
        ax, ay, az = transforms3d.euler.mat2euler(self.init_robot_pose[:3, :3], self.robot.trans_method)
        euler = [ax, ay, az]
        print("----------------------------")
        print(euler)
        print("----------------------------")

        # 再生成4组robot_pose,让机器人移动到此位置,拍照并计算外参
        for i in [1,2]:
            for j in [-2,2]:
                objpoint_temp = None
                imgpoint_temp = None
                robot_pose_temp = None
                euler_temp = euler.copy()
                rgb_image_temp = None
                depth_image_temp = None
                robot_pose_str_temp = None
                capture_times = 0
                while (True):
                    print("i,j:",i,j)
                    print("capture_times", capture_times)
                    # 如果重拍次数超过3次，但是仍然检测不到角点（或者是未检测全），则放弃该位置
                    if capture_times >= 3:
                        break
                    if self.cali_type == 0:
                        # print('0')
                        euler_temp[i] += j * math.pi / 60   #60
                    else:
                        # print('type1 angle',j * math.pi / 12/math.pi*180)
                        euler_temp[i] += j * math.pi / 15   #12
                    print(self.robot.trans_method)
                    pose_r = transforms3d.euler.euler2mat(euler_temp[0], euler_temp[1], euler_temp[2], self.robot.trans_method)
                    robot_pose = self.init_robot_pose.copy()
                    robot_pose[:3, :3] = pose_r[:, :]
                    robot_pose[0][3]+=random.uniform(-0.1,0.1)
                    robot_pose[1][3]+=random.uniform(-0.1,0.1)
                    robot_pose[2][3]+=random.uniform(0,0.1)
                    # print('get euler',euler_temp[0]*180/3.14159,euler_temp[1]*180/3.14159,euler_temp[2]*180/3.14159)
                    # print('robot pose',robot_pose)

                    # import pdb
                    # pdb.set_trace()

                    flag = self.robot.moveable(robot_pose,1)
                    if not flag:
                        break
                    # print(flag)
                    flag, robot_pose, robot_pose_str = self.robot.move_mat(robot_pose)
                    if not flag:
                        break
                    # print(flag)
                    if self.camera.type=='3d':
                        flag, rgb, depth, img_RootPath = self.camera.get_rgb_depth_image()
                    else:
                        flag, rgb, img_RootPath = self.camera.get_rgb_image()
                    capture_times += 1
                    # rgb = cv2.undistort(rgb, self.camera.intrinsic, self.camera.dist)
                    flag, objpoint, imgpoint = self.board.getObjImgPointList(rgb,0)
                    # circle
                    # flag, objpoint, imgpoint = self.board.getObjImgPointListFromCircle(rgb, 5, 4)
                    print('detect flag',flag)
                    if flag:
                        objpoint_temp = objpoint.copy()
                        imgpoint_temp = imgpoint.copy()
                        robot_pose_temp = robot_pose.copy()
                        rgb_image_temp = rgb.copy()
                        if self.camera.type=='3d':
                            depth_image_temp = depth.copy()
                        robot_pose_str_temp = robot_pose_str
                        break
                    else:
                        data = {'DepthMap': [], 'PointCloudZ': []}
                        data_path, imgName_list, robot_pose = load_data.load_data(img_RootPath, data, imgName_list=[], robot_pose=[])
                        os.remove(imgName_list[-1])
                        os.remove(data_path['PointCloudZ'][-1])

                if not objpoint_temp is None:
                    if self.camera.type=='3d':
                        flag, camerapose, depthcorner, markercorner = getCameraposeFromDepth(self.board, rgb, depth,self.camera.intrinsic,self.camera.dist)
                        self.depth_image.append(depth_image_temp)
                        self.depth_corner.append(depthcorner)
                        self.marker_corner.append(markercorner)
                        print('camerapose svd', camerapose)
                        camerapose = self.board.extrinsic(imgpoint, objpoint, self.camera.intrinsic, self.camera.dist)
                        print('camerapose pnp', camerapose)
                    else:
                        camerapose = self.board.extrinsic(imgpoint, objpoint, self.camera.intrinsic, self.camera.dist)
                    self.Hobj2camera.append(camerapose)
                    self.objpoint_list.append(objpoint_temp)
                    self.imgpoint_list.append(imgpoint_temp)
                    self.Hend2base.append(robot_pose_temp)
                    self.rgb_image.append(rgb_image_temp)
                    self.save_txt.append(robot_pose_str_temp)

        assert len(self.Hend2base) > 3, "没有得到足够有效的初始数据！请更换初始位置！"
        print('len img',len(self.rgb_image))
        return
    def semi_init_handeye(self,posetxtpath):
        '''
        通过旋转机械臂来初始化手眼标定，得到较为粗略的手眼矩阵
        :return:
        '''
        # 初次获取robot_pose,得到一组数据,计算出来外参
        # angle
        flag, robot_pose_str = self.robot.init_move(self.init_robot_pose_str)
        assert flag, "该位姿机器人不可达!"
        if self.camera.type=='3d':
            print('3d')
            flag, rgb, depth, img_RootPath = self.camera.get_rgb_depth_image()
        else:
            print('2d')
            flag, rgb , img_RootPath= self.camera.get_rgb_image()
        assert flag, "拍摄失败!"
        flag, objpoint, imgpoint = self.board.getObjImgPointList(rgb)
        print('intr and dist',self.camera.intrinsic,self.camera.dist)
        if not flag:
            str1='msg|该位姿下标定板角点无法被拍摄到,please adjust board location!'
            self.camera.sendProcessBar(str1)
        assert flag, "该位姿下标定板角点无法被拍摄到!"

        if self.camera.type=='3d':
            flag, camerapose, depthcorner, markercorner = getCameraposeFromDepth(self.board, rgb, depth,self.camera.intrinsic, self.camera.dist)
            self.depth_image = []
            self.depth_corner = []  # get corner in camera coordidate from depth and camera intrinsic
            self.marker_corner = []  # get corner in world coordidate from real marker size
            self.depth_image.append(depth)
            self.depth_corner.append(depthcorner)
            self.marker_corner.append(markercorner)
            print('camerapose svd')
            # print(flag)
            print(camerapose)

            camerapose = self.board.extrinsic(imgpoint, objpoint, self.camera.intrinsic, self.camera.dist)
            print('camerapose pnp', camerapose)
        else:
            camerapose = self.board.extrinsic(imgpoint, objpoint, self.camera.intrinsic, self.camera.dist)
        # assert flag, "获取相机外参失败!"
        self.objpoint_list = []
        self.imgpoint_list = []
        self.Hend2base = []
        self.Hobj2camera = []
        self.rgb_image = []
        self.result = []
        self.depth_corner=[]  #get corner in camera coordidate from depth and camera intrinsic
        self.marker_corner=[] #get corner in world coordidate from real marker size
        self.objpoint_list.append(objpoint)
        self.imgpoint_list.append(imgpoint)
        self.rgb_image.append(rgb)
        self.Hend2base.append(self.init_robot_pose)
        self.save_txt.append(robot_pose_str)
        self.Hobj2camera.append(camerapose)
        # self.robot.trans_method = ''
        ax, ay, az = transforms3d.euler.mat2euler(self.init_robot_pose[:3, :3], self.robot.trans_method)
        euler = [ax, ay, az]
        print("----------------------------")
        print(euler)
        print("----------------------------")

        data = []
        posetxtpath = posetxtpath + '/robotpose.txt'
        print('semi robot pose path:',posetxtpath)
        for line in open(posetxtpath, "r"):  # 设置文件对象并读取每一行文件
            data.append(line)  # 将每一行文件加入到list中
        indexpro=0
        for strline in data:
            pbarstr = "externsic|ProgressBar|正在标定，已完成" + str(round( indexpro / len(data) * 100.0, 2)) + "%。"
            self.camera.sendProcessBar(pbarstr)
            indexpro += 1
            print('read txt ')
            print(strline)
            res = []
            for value in strline.split(','):
                res.append(float(value))
            objpoint_temp = None
            imgpoint_temp = None
            robot_pose_temp = None
            rgb_image_temp = None
            depth_image_temp = None
            robot_pose_str_temp = None
            print('robot pose from txt :',res[3],res[4],res[5],res[0],res[1],res[2])
            pose_r = transforms3d.euler.euler2mat(res[3]*3.1416/180, res[4]*3.1416/180, res[5]*3.1416/180, self.robot.trans_method)
            robot_pose = self.init_robot_pose.copy()
            robot_pose[:3, :3] = pose_r[:, :]
            robot_pose[0][3] = res[0]/1000
            robot_pose[1][3] = res[1]/1000
            robot_pose[2][3] = res[2]/1000
            # print('robot pose',robot_pose)

            flag = self.robot.moveable(robot_pose,0)
            # print(flag)
            flag, robot_pose, robot_pose_str = self.robot.move_mat(robot_pose)
            if not flag:
                print('robot move_mat return false')
                break
            if self.camera.type=='3d':
                flag, rgb, depth, img_RootPath = self.camera.get_rgb_depth_image()
            else:
                flag, rgb, img_RootPath = self.camera.get_rgb_image()
            # rgb = cv2.undistort(rgb, self.camera.intrinsic, self.camera.dist)
            flag, objpoint, imgpoint = self.board.getObjImgPointList(rgb,0)
            # circle
            # flag, objpoint, imgpoint = self.board.getObjImgPointListFromCircle(rgb, 5, 4)
            print('detect flag',flag)
            if flag:
                objpoint_temp = objpoint.copy()
                imgpoint_temp = imgpoint.copy()
                robot_pose_temp = robot_pose.copy()
                rgb_image_temp = rgb.copy()
                if self.camera.type=='3d':
                    depth_image_temp = depth.copy()
                robot_pose_str_temp = robot_pose_str
            else:
                data = {'DepthMap': [], 'PointCloudZ': []}
                data_path, imgName_list, robot_pose = load_data.load_data(img_RootPath, data, imgName_list=[], robot_pose=[])
                os.remove(imgName_list[-1])
                # os.remove(data_path['PointCloudZ'][-1])

            if not objpoint_temp is None:
                if self.camera.type=='3d':
                    flag, camerapose, depthcorner, markercorner = getCameraposeFromDepth(self.board, rgb, depth,self.camera.intrinsic,self.camera.dist)
                    self.depth_image.append(depth_image_temp)
                    self.depth_corner.append(depthcorner)
                    self.marker_corner.append(markercorner)
                    print('camerapose svd', camerapose)
                    camerapose = self.board.extrinsic(imgpoint, objpoint, self.camera.intrinsic, self.camera.dist)
                    print('camerapose pnp', camerapose)
                else:
                    camerapose = self.board.extrinsic(imgpoint, objpoint, self.camera.intrinsic, self.camera.dist)
                self.Hobj2camera.append(camerapose)
                self.objpoint_list.append(objpoint_temp)
                self.imgpoint_list.append(imgpoint_temp)
                self.Hend2base.append(robot_pose_temp)
                self.rgb_image.append(rgb_image_temp)
                self.save_txt.append(robot_pose_str_temp)

        assert len(self.Hend2base) > 3, "没有得到足够有效的初始数据！请更换初始位置！"
        print('len img',len(self.rgb_image))
        return

    def handeye_cali(self):
        if self.cali_type == 0:
            from handineye_org import motion
            from handineye_org import rx
            from handineye_org import rz
        else:
            from handtoeye_org import motion
            from handtoeye_org import rx
            from handtoeye_org import rz
        # A, B = motion.motion_axxb(self.Hend2base, self.Hobj2camera)
        A, B = motion.motion_axyb(self.Hend2base, self.Hobj2camera)
        # Hx,Hy = li.calibration(A,B)
        # Hx = dual.calibration(A, B)
        Hx = tsai.calibration(A, B)
        Hx = rx.refine(Hx, self.Hend2base, self.Hobj2camera,  self.board.GetBoardAllPoints())
        q = np.array([])
        t = np.array([])
        for i in range(len(self.Hobj2camera)):
            if self.cali_type == 0:
                Hy = np.dot(self.Hend2base[i], np.dot(Hx, self.Hobj2camera[i]))
            else:
                Hy = np.dot(np.linalg.inv(self.Hend2base[i]), np.dot(Hx, self.Hobj2camera[i]))
            q_temp = transforms3d.quaternions.mat2quat(Hy[:3, :3])

            if i == 0:
                q0 = q_temp.copy()
            else:
                if np.linalg.norm(q0 - q_temp) > np.linalg.norm(q0 + q_temp):
                    q_temp = -q_temp
            q = np.append(q, q_temp)
            t = np.append(t, Hy[:3, 3])
        q = q.reshape([-1, 4])
        t = t.reshape([-1, 3])
        q_mean = np.mean(q, 0)
        t_mean = np.mean(t, 0)
        q = q_mean / np.linalg.norm(q)
        Hy_r = transforms3d.quaternions.quat2mat(q)
        Hy = np.identity(4)
        Hy[:3, :3] = Hy_r[:, :]
        Hy[:3, 3] = t_mean[:]
        # print("come in rme")
        rme = rz.proj_error(Hx, Hy, self.Hend2base, self.Hobj2camera, self.board.GetBoardAllPoints())

        Hxli,Hyli = li.calibration(A,B)
        rmeli = rz.proj_error(Hxli, Hyli, self.Hend2base, self.Hobj2camera, self.board.GetBoardAllPoints())
        print('hxli',Hxli)
        print('li error',np.mean(np.abs(rmeli)))
        print('hx tsai',Hx)
        print('tsai error',np.mean(np.abs(rme)))
        # rme1 = rz.proj_error1(Hx, Hy, self.Hend2base, self.depth_corner, self.marker_corner)
        # print("-----------------------------------------")
        # print("the projection mean error is:" + str(rme))
        # print("mean",np.mean(np.abs(rme1)))

        if self.cali_type == 0:
            self.result.append({"image_number": len(self.rgb_image), "Hcamera2end": Hx,"Hobj2base": Hy,
                                 "mean_rme": np.mean(np.abs(rmeli)), "max_rme": np.max(np.abs(rmeli))})
        else:
            self.result.append({"image_number": len(self.rgb_image), "Hcamera2base": Hx, "Hobj2end": Hy,
                                "mean_rme": np.mean(np.abs(rmeli)), "max_rme": np.max(np.abs(rmeli))})

        if (np.mean(np.abs(rmeli))) < 0.003 or np.mean(np.abs(rme)) < 0.003  or len(self.rgb_image) < 6 :
            if(np.mean(np.abs(rmeli))<np.mean(np.abs(rme)) or (np.mean(np.abs(rmeli))< 0.0015 and np.mean(np.abs(rme)) > 0.001)):
                self.Hx = Hxli
                self.Hy = Hyli
                self.error = np.mean(np.abs(rmeli))
            else:
                self.Hx = Hx
                self.Hy = Hy
                self.error = np.mean(np.abs(rme))
            print("final hx:")
            print(self.Hx)
            # print("mean:", np.mean(np.abs(rme)))
            # print("meanli:", np.mean(np.abs(rmeli)))
            return True
        else:
            del self.objpoint_list[-1]
            del self.imgpoint_list[-1]
            del self.Hend2base[-1]
            del self.Hobj2camera[-1]
            del self.rgb_image[-1]
            if self.camera.type=='3d':
                print('len depth img',len(self.depth_image))
                del self.depth_image[-1]
            print("delete this image,hx before is:")
            print(self.Hx)
            return False
    def handeye_cali2(self):
        if self.cali_type == 0:
            from handineye_org import motion
            from handineye_org import rx
            from handineye_org import rz
        else:
            from handtoeye_org import motion
            from handtoeye_org import rx
            from handtoeye_org import rz
        # A, B = motion.motion_axxb(self.Hend2base, self.Hobj2camera)
        A, B = motion.motion_axyb(self.Hend2base, self.Hobj2camera)
        # Hx,Hy = li.calibration(A,B)
        # Hx = dual.calibration(A, B)
        Hx = tsai.calibration(A, B)
        Hx = rx.refine(Hx, self.Hend2base, self.Hobj2camera,  self.board.GetBoardAllPoints())
        q = np.array([])
        t = np.array([])
        for i in range(len(self.Hobj2camera)):
            if self.cali_type == 0:
                Hy = np.dot(self.Hend2base[i], np.dot(Hx, self.Hobj2camera[i]))
            else:
                Hy = np.dot(np.linalg.inv(self.Hend2base[i]), np.dot(Hx, self.Hobj2camera[i]))
            q_temp = transforms3d.quaternions.mat2quat(Hy[:3, :3])

            if i == 0:
                q0 = q_temp.copy()
            else:
                if np.linalg.norm(q0 - q_temp) > np.linalg.norm(q0 + q_temp):
                    q_temp = -q_temp
            q = np.append(q, q_temp)
            t = np.append(t, Hy[:3, 3])
        q = q.reshape([-1, 4])
        t = t.reshape([-1, 3])
        q_mean = np.mean(q, 0)
        t_mean = np.mean(t, 0)
        q = q_mean / np.linalg.norm(q)
        Hy_r = transforms3d.quaternions.quat2mat(q)
        Hy = np.identity(4)
        Hy[:3, :3] = Hy_r[:, :]
        Hy[:3, 3] = t_mean[:]
        # print("come in rme")
        rme = rz.proj_error(Hx, Hy, self.Hend2base, self.Hobj2camera, self.board.GetBoardAllPoints())

        Hxli,Hyli = li.calibration(A,B)
        rmeli = rz.proj_error(Hxli, Hyli, self.Hend2base, self.Hobj2camera, self.board.GetBoardAllPoints())
        print('hxli',Hxli)
        print('li error',np.mean(np.abs(rmeli)))
        print('hx tsai',Hx)
        print('tsai error',np.mean(np.abs(rme)))

        if self.cali_type == 0:
            self.result.append({"image_number": len(self.rgb_image), "Hcamera2end": Hx,"Hobj2base": Hy,
                                 "mean_rme": np.mean(np.abs(rmeli)), "max_rme": np.max(np.abs(rmeli))})
        else:
            self.result.append({"image_number": len(self.rgb_image), "Hcamera2base": Hx, "Hobj2end": Hy,
                                "mean_rme": np.mean(np.abs(rmeli)), "max_rme": np.max(np.abs(rmeli))})

        if True :
            if(np.mean(np.abs(rmeli))<np.mean(np.abs(rme))):
                self.Hx = Hxli
                self.Hy = Hyli
                self.error = np.mean(np.abs(rmeli))
            else:
                self.Hx = Hx
                self.Hy = Hy
                self.error = np.mean(np.abs(rme))
            print("final hx:")
            print(self.Hx)
            # print("mean:", np.mean(np.abs(rme)))
            # print("meanli:", np.mean(np.abs(rmeli)))
            return True
    def semi_handeye_cali(self):
        if self.cali_type == 0:
            from handineye_org import motion
            from handineye_org import rx
            from handineye_org import rz
        else:
            from handtoeye_org import motion
            from handtoeye_org import rx
            from handtoeye_org import rz
        # A, B = motion.motion_axxb(self.Hend2base, self.Hobj2camera)
        while True:
            A, B = motion.motion_axyb(self.Hend2base, self.Hobj2camera)
            print('len data',len(self.Hend2base),len(self.Hobj2camera))
            # Hx,Hy = li.calibration(A,B)
            # Hx = dual.calibration(A, B)
            Hx = tsai.calibration(A, B)
            Hx = rx.refine(Hx, self.Hend2base, self.Hobj2camera,  self.board.GetBoardAllPoints())
            q = np.array([])
            t = np.array([])
            for i in range(len(self.Hobj2camera)):
                if self.cali_type == 0:
                    Hy = np.dot(self.Hend2base[i], np.dot(Hx, self.Hobj2camera[i]))
                else:
                    Hy = np.dot(np.linalg.inv(self.Hend2base[i]), np.dot(Hx, self.Hobj2camera[i]))
                q_temp = transforms3d.quaternions.mat2quat(Hy[:3, :3])

                if i == 0:
                    q0 = q_temp.copy()
                else:
                    if np.linalg.norm(q0 - q_temp) > np.linalg.norm(q0 + q_temp):
                        q_temp = -q_temp
                q = np.append(q, q_temp)
                t = np.append(t, Hy[:3, 3])
            q = q.reshape([-1, 4])
            t = t.reshape([-1, 3])
            q_mean = np.mean(q, 0)
            t_mean = np.mean(t, 0)
            q = q_mean / np.linalg.norm(q)
            Hy_r = transforms3d.quaternions.quat2mat(q)
            Hy = np.identity(4)
            Hy[:3, :3] = Hy_r[:, :]
            Hy[:3, 3] = t_mean[:]
            # print("come in rme")
            rme = rz.proj_error(Hx, Hy, self.Hend2base, self.Hobj2camera, self.board.GetBoardAllPoints())
            rmedelete = rz.proj_error2(Hx, Hy, self.Hend2base, self.Hobj2camera, self.board.GetBoardAllPoints())
            Hxli,Hyli = li.calibration(A,B)
            rmeli = rz.proj_error(Hxli, Hyli, self.Hend2base, self.Hobj2camera, self.board.GetBoardAllPoints())
            print('hxli',Hxli)
            print('li error',np.mean(np.abs(rmeli)))
            print('hx tsai',Hx)
            print('tsai error',np.mean(np.abs(rme)))
            # rme1 = rz.proj_error1(Hx, Hy, self.Hend2base, self.depth_corner, self.marker_corner)
            # print("-----------------------------------------")
            # print("the projection mean error is:" + str(rme))
            # print("mean",np.mean(np.abs(rme1)))
            print(rmedelete)
            print('rme shape',rmedelete.shape)
            x,y = np.where(rmedelete.reshape([1,-1]) == np.max(rmedelete))
            if len(self.Hobj2camera) < 9 or np.mean(np.abs(rme)) < 0.002:
                print('over cali')
                print(y.shape[0])
                break
            print(y[0])
            print(x[0])
            del self.Hend2base[y[0]]
            del self.Hobj2camera[y[0]]
        if self.cali_type == 0:
            self.result.append({"image_number": len(self.rgb_image), "Hcamera2end": Hx,"Hobj2base": Hy,
                                 "mean_rme": np.mean(np.abs(rmeli)), "max_rme": np.max(np.abs(rmeli))})
        else:
            self.result.append({"image_number": len(self.rgb_image), "Hcamera2base": Hx, "Hobj2end": Hy,
                                "mean_rme": np.mean(np.abs(rmeli)), "max_rme": np.max(np.abs(rmeli))})
        self.Hx = Hx
        self.Hy = Hy
        self.error = np.mean(np.abs(rme))
        print("final hx:")
        print(self.Hx)
        print("mean:", np.mean(np.abs(rme)))
        # print("meanli:", np.mean(np.abs(rmeli)))
        return True

    def camera_pose_simple(self, verbose=0):
        '''
        :param verbose:
        :return: 返回所有可能的外参，以供挑选
        '''
        def getBaseCampose(initial_rotation):
            '''
            :param initial_rotation: 初始（板->相机）的旋转矩阵
            :return:该位置下最优的（板->相机）旋转矩阵
            '''
            a_ = math.pi / 2
            b_ = math.pi / 2
            min_z = 2
            # 找到使min_z最小的角度
            # min_z是p[2, 2]与1的距离
            for i in range(10):
                a = math.pi / 20 * i
                for j in range(10):
                    b = math.pi / 20 * j
                    rotation = transforms3d.euler.euler2mat(a, b, 0)
                    p = np.dot(rotation, initial_rotation)
                    dis_z = abs(p[2, 2] - 1)
                    if p[2, 2] > 0.9:
                        min_z = 0
                        if abs(a) + abs(b) < abs(a_) + abs(b_):
                            a_ = a
                            b_ = b
                    else:
                        if dis_z < min_z:
                            a_ = a
                            b_ = b
                            min_z = dis_z
            # 如果min_z还是很大，那么便优化出该点下p[2,2]最接近1的旋转角度
            if min_z > 0.2:
                def loss(X, initial_rotation):
                    a = X[0]
                    b = X[1]
                    rotation = transforms3d.euler.euler2mat(a, b, 0)
                    p = np.dot(rotation, initial_rotation)
                    return abs(p[2, 2] - 1)
                init = np.array([0, 0])
                solver = opt.minimize(loss, init, initial_rotation)
                X = solver.x
                a_ = X[0]
                b_ = X[1]
            rotation = transforms3d.euler.euler2mat(a_, b_, 0)
            p = np.dot(rotation, initial_rotation)
            return p

        z_cout = int((self.maxZ - self.minZ) / self.inter_z)
        sample_position = np.array([])
        for i in range(z_cout):
            z = self.minZ + i * self.inter_z
            extent = z * math.tan(self.optic_angle / 180.0 * math.pi)
            max_x = extent
            min_x = -extent
            max_y = extent
            min_y = -extent
            # 对空间的点进行采样
            position = np.mgrid[min_x: max_x: self.inter_xy, min_y: max_y: self.inter_xy].T.reshape(-1, 2)
            if self.cali_type == 0:
                sample_position = np.append(sample_position,
                                            np.append(position, np.dot(-z, np.ones([position.shape[0], 1])), 1))
            else:
                sample_position = np.append(sample_position,
                                            np.append(position, np.dot(z, np.ones([position.shape[0], 1])), 1))

        # 得到全部的采样位置（板->相机的转换），供下一步挑选
        print(1.1)
        sample_position = sample_position.reshape([-1, 3])

        # 画图
        if verbose == 1:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.scatter3D(sample_position[:, 0], sample_position[:, 1], sample_position[:, 2], cmap='Blues')
            # ax.scatter3D([0, 0, board_max_x, board_max_x], [0, board_max_y, 0, board_max_y], [0, 0, 0, 0])
            #plt.show()
            plt.savefig("caiyang.png")
            #os.system("pause")

        # 得到初始位置中相机的最优姿态
        print(1.2)
        init_base_camepose = getBaseCampose(self.Hobj2camera[0][:3, :3])
        ax, ay, az = transforms3d.euler.mat2euler(init_base_camepose)

        if self.cali_type == 0:
            angle = np.array([[0, 0, 0],
                              [math.pi / 6, 0, 0],
                              [math.pi / 3, 0, 0],
                              [0, math.pi / 6, 0],
                              [0, math.pi / 3, 0],
                              [0, 0, math.pi / 6],
                              [0, 0, math.pi / 3],
                              [-math.pi / 6, 0, 0],
                              [-math.pi / 3, 0, 0],
                              [0, -math.pi / 6, 0],
                              [0, -math.pi / 3, 0],
                              [0, 0, -math.pi / 6],
                              [0, 0, -math.pi / 3],
                              ])
        else:
            angle = np.array([[0, 0, 0],
                              [math.pi / 9,0,0],   #15   #30
                              [math.pi / 4, 0, 0],
                              [0, math.pi / 9, 0],
                              [0, math.pi / 4, 0],
                              [0, 0, math.pi / 9],
                              [0, 0, math.pi / 4],
                              [-math.pi / 9, 0, 0],
                              [-math.pi / 4, 0, 0],
                              [0, -math.pi / 9, 0],
                              [0, -math.pi / 4, 0],
                              [0, 0, -math.pi / 9],
                              [0, 0, -math.pi / 4],
                              ])

        Rlist = []
        for i in range(angle.shape[0]):
            R = transforms3d.euler.euler2mat(angle[i, 0], angle[i, 1], angle[i, 2])
            Rlist.append(np.dot(R, init_base_camepose))
        sample_cam_pose = []
        print(1.3)
        for i in range(sample_position.shape[0]):
            for R in Rlist:
                if self.cali_type == 0:
                    H = np.append(np.append(np.linalg.inv(R), np.transpose([sample_position[i, :]]), 1),np.array([[0, 0, 0, 1]]), 0)
                    sample_cam_pose.append(np.linalg.inv(H))
                else:
                    H = np.append(np.append(R, np.transpose([sample_position[i, :]]), 1),np.array([[0, 0, 0, 1]]), 0)
                    sample_cam_pose.append(H)
        print(1.4)
        return sample_cam_pose

    def select_pose_by_view(self, sample_cam_pose):
        print(2.1)
        board_max_x = self.board.marker_X * (self.board.markerSeparation + self.board.tag_size)
        board_max_y = self.board.marker_Y * (self.board.markerSeparation + self.board.tag_size)
        mid_x = board_max_x / 2
        mid_y = board_max_y / 2

        # marker板的四个边界点
        mid_points = np.array([[0, mid_x, mid_x, board_max_x],
                               [mid_y, 0, board_max_y, mid_y],
                               [0, 0, 0, 0],
                               [1, 1, 1, 1]])
        select_pose = []
        all = 0
        print(2.2)
        camera_intrinsic = np.append(self.camera.intrinsic, np.zeros([3, 1]), 1)
        imgsize = self.camera.imgsize
        for i in range(len(sample_cam_pose)):
            camera_pose = sample_cam_pose[i]
            points = np.dot(camera_intrinsic, np.dot(camera_pose, mid_points))

            if (points[2, 0] < 0):
                continue
            points[:, :] = points[:, :] / points[2, :]
            t = 0
            # 记录在照片中的点的个数
            for j in range(4):
                if points[0, j] > 0 and points[0, j] < imgsize[0] and points[1, j] > 0 and points[1, j] < imgsize[1]:
                    t = t + 1
            if t > 3 :#t > 3
                # print("imgsize:",imgsize[0],imgsize[1])
                # for j in range(4):
                #     print(points[0, j],points[1, j]) 
                select_pose.append(camera_pose)
            if t == 4:
                all = all + 1

        return select_pose

    def score_std(self,campose):
        expect_cam_pose_mat = campose

        q = np.array([])
        t = np.array([])
        for j in range(len(self.Hend2base)):
            if self.cali_type == 0:
                temp_robot_pose = np.dot(self.Hend2base[j], np.dot(self.Hx, np.dot(self.Hobj2camera[j], np.dot(
                    np.linalg.inv(expect_cam_pose_mat), np.linalg.inv(self.Hx)))))
            else:
                temp_robot_pose = np.dot(self.Hx, np.dot(campose, np.dot(np.linalg.inv(self.Hobj2camera[j]),
                                                       np.dot(np.linalg.inv(self.Hx), self.Hend2base[j]))))
            q = np.append(q, transforms3d.quaternions.mat2quat(temp_robot_pose[:3, :3]))
            t = np.append(t, temp_robot_pose[:3, 3])
        q = q.reshape([-1, 4])
        t = t.reshape([-1, 3])
        for i in range(1, q.shape[0]):
            if abs(np.linalg.norm(q[0, :] - q[i, :])) > abs(np.linalg.norm(q[0, :] + q[i, :])):
                q[i, :] = -q[i, :]
        mean_q = np.mean(q, 0)
        mean_t = np.mean(t, 0)
        std_q = np.std(q, axis=0)
        std_t = np.std(t, axis=0)

        expect_robot_pose = np.append(transforms3d.quaternions.quat2mat(mean_q), np.transpose([mean_t]), 1)
        expect_robot_pose = np.append(expect_robot_pose, np.array([[0, 0, 0, 1]]), 0)
        score = np.sum(std_q) + np.sum(std_t)
        return score, expect_robot_pose

    def get_Expect_robot_pose(self, expect_campose):
        '''
        中间转化为四元数进行运算
        :param expect_campose: 期望的相机位姿
        :return: 该相机位姿对应的robot_pose(matrix)
        '''
        expect_cam_pose_mat = expect_campose
        q = np.array([])
        t = np.array([])
        for j in range(len(self.Hend2base)):
            if self.cali_type == 0:
                temp_robot_pose = np.dot(self.Hend2base[j], np.dot(self.Hx, np.dot(self.Hobj2camera[j], np.dot(
                    np.linalg.inv(expect_cam_pose_mat), np.linalg.inv(self.Hx)))))
            else:
                temp_robot_pose = np.linalg.multi_dot([self.Hx, expect_cam_pose_mat, np.linalg.inv(self.Hobj2camera[j]),
                                                       np.linalg.inv(self.Hx), self.Hend2base[j]])
            q = np.append(q, transforms3d.quaternions.mat2quat(temp_robot_pose[:3, :3]))
            t = np.append(t, temp_robot_pose[:3, 3])
        q = q.reshape([-1, 4])
        t = t.reshape([-1, 3])
        # 末端->板的转换按理来说是固定的，取平均值来减小误差
        for i in range(1, q.shape[0]):
            if abs(np.linalg.norm(q[0, :] - q[i, :])) > abs(np.linalg.norm(q[0, :] + q[i, :])):   #???
                q[i, :] = -q[i, :]
        #print("T")
        #print(t)
        mean_q = np.mean(q, 0)
        mean_t = np.mean(t, 0) 
        #print("meanT")
        #print(mean_t)
        expect_robot_pose = np.append(transforms3d.quaternions.quat2mat(mean_q), np.transpose([mean_t]), 1)
        expect_robot_pose = np.append(expect_robot_pose, np.array([[0, 0, 0, 1]]), 0)

        #new get robot pose
        temp_except_robot_pose=np.linalg.multi_dot([self.Hx,expect_cam_pose_mat,np.linalg.inv(self.Hy)])
        my_except_robot_pose=np.linalg.inv(temp_except_robot_pose)

        # print("expect_robot_pose")
        # print(expect_robot_pose)
        # print("my_except_robot_pose")
        # print(my_except_robot_pose)   这个结果不靠谱

        return expect_robot_pose

    def score_no_local(self, expect_robot_pose):
        q0 = transforms3d.quaternions.mat2quat(expect_robot_pose[:3, :3])
        t0 = expect_robot_pose[:3, 3]
        min_score = 0
        for j in range(len(self.Hend2base)):
            q = transforms3d.quaternions.mat2quat(self.Hend2base[j][:3, :3])
            if np.linalg.norm(q - q0) > np.linalg.norm(q + q0):
                q = -q
            t = self.Hend2base[j][:3, 3]
            q_dis = np.linalg.norm(q - q0)
            t_dis = np.linalg.norm(t - t0)
            score = -(math.pow(math.e, -0.5*abs(q_dis)) + 1 * math.pow(math.e, -0.5*abs(t_dis)))
            if score < min_score:
                min_score = score
        return min_score

    def score_expect_rme(self, campose):
        if self.cali_type == 0:
            from handineye_org import motion
            from handineye_org import rx
            from handineye_org import rz
        else:
            from handtoeye_org import motion
            from handtoeye_org import rx
            from handtoeye_org import rz
        robot_pose = self.get_Expect_robot_pose(campose)
        rme = rz.proj_error(self.Hx, self.Hy, [robot_pose], [campose], self.board.GetBoardAllPoints())
        return np.max(np.abs(rme)), robot_pose

    def score_main(self, expect_camera_list):
        time1 = time.time()
        if self.next_step_method == 1 or self.next_step_method == 3:
            sco_list = []
            for i in range(len(expect_camera_list)):
                score, robot_pose = self.score_std(expect_camera_list[i])
                if self.next_step_method == 3:
                    no_local_score = self.score_no_local(robot_pose)
                    score = score * 10 + no_local_score
                sco_list.append([i, score, expect_camera_list[i]])
            time2 = time.time()
            print("score time:", time2 - time1, "len(expect_camera_list)=", len(expect_camera_list))
            sco_list.sort(key=lambda x: x[1])
            campose_order_list = []
            for t in sco_list:
                campose_order_list.append(t[2])
            campose_order_list.reverse()
            time3 = time.time()
            print("sort time:", time3-time2)
            return campose_order_list
        elif self.next_step_method == 2 or self.next_step_method == 4:
            sco_list = []
            for i in range(len(expect_camera_list)):
                score, robot_pose = self.score_expect_rme(expect_camera_list[i])
                if not self.robot.moveable(robot_pose):
                    continue
                if self.next_step_method == 4:
                    no_local_score = self.score_no_local(robot_pose)
                    score = score * (10 ** 5) + no_local_score
                sco_list.append([i, score, expect_camera_list[i]])
            sco_list.sort(key=lambda x: x[1])
            campose_order_list = []
            for t in sco_list:
                campose_order_list.append(t[2])
            campose_order_list.reverse()
            return campose_order_list
        elif self.next_step_method == 0:
            sco_list = []
            for i in range(len(expect_camera_list)):
                robot_pose = self.get_Expect_robot_pose(expect_camera_list[i])
                no_local_score = self.score_no_local(robot_pose)
                sco_list.append([i, no_local_score, expect_camera_list[i]])
            sco_list.sort(key=lambda x: x[1])
            campose_order_list = []
            for t in sco_list:
                campose_order_list.append(t[2])
            campose_order_list.reverse()
            return campose_order_list
        else:
            random.shuffle(expect_camera_list)
            return expect_camera_list

    def score_main_multi(self, expect_camera_list):
        # if self.next_step_method == 5:
        #     random.shuffle(expect_camera_list)
        #     return expect_camera_list
        # else:
        #     campose_order_list = multi_score(self.cali_type, self.Hend2base, self.Hobj2camera, self.next_step_method,
        #                                      self.Hx, expect_camera_list)
        #     return campose_order_list

        # campose_order_list = multi_score(self.cali_type, self.Hend2base, self.Hobj2camera, self.next_step_method,
        #                                  self.Hx, expect_camera_list)
        # return campose_order_list

        random.shuffle(expect_camera_list)
        return expect_camera_list
        


    def copy(self):
        auto = auto_handeye_calibration(self.board, self.robot, self.camera, self.config)
        auto.cali_type = self.cali_type
        auto.next_step_method = self.next_step_method
        auto.Hx = self.Hx.copy()
        auto.Hy = self.Hy.copy()
        auto.imgpoint_list = self.imgpoint_list.copy()
        auto.objpoint_list = self.objpoint_list.copy()
        auto.Hend2base = self.Hend2base.copy()
        auto.Hobj2camera = self.Hobj2camera.copy()
        auto.rgb_image = self.rgb_image.copy()
        auto.depth_image = self.depth_image.copy()
        auto.result = self.result.copy()
        return auto

    def ias_run(self):
        num_p = 3
        x_angle = [0, 15, 0]
        y_angle = [0, 0, 15]
        z_angle = [0, 0, 0]
        d_min = -0.4
        widgets = ['ias: ', Percentage(), ' ', Bar('#'), ' ', Timer(),
                   ' ', ETA(), ]
        pbar = ProgressBar(widgets=widgets, maxval=30).start()
        for plane in range(num_p):
            for angle in range(len(x_angle)):
                Hcamera2obj = np.linalg.inv(self.Hobj2camera[0])
                r_x_temp, r_y_temp, r_z_temp = transforms3d.euler.mat2euler(Hcamera2obj[:3, :3])
                r_x_temp = r_x_temp + x_angle[angle] * math.pi / 180
                r_y_temp = r_y_temp + y_angle[angle] * math.pi / 180
                r_z_temp = r_z_temp + z_angle[angle] * math.pi / 180
                for y in range(4):
                    tx_temp, ty_temp, tz_temp = Hcamera2obj[:3, 3]
                    while (True):
                        if y == 0:
                            tx_temp = tx_temp + 0.05
                        elif y == 1:
                            tx_temp = tx_temp - 0.05
                        elif y == 2:
                            ty_temp = ty_temp + 0.05
                        elif y == 3:
                            ty_temp = ty_temp - 0.05
                        expect_camera2obj_r = transforms3d.euler.euler2mat(r_x_temp, r_y_temp, r_z_temp)
                        expect_camera2obj = np.identity(4)
                        expect_camera2obj[:3, :3] = expect_camera2obj_r
                        expect_camera2obj[0, 3] = tx_temp
                        expect_camera2obj[1, 3] = ty_temp
                        expect_camera2obj[2, 3] = d_min
                        expect_robot_pose = self.get_Expect_robot_pose(np.linalg.inv(expect_camera2obj))
                        flag = self.robot.moveable(expect_robot_pose)
                        if not flag:
                            break
                        flag, rgb_image = self.move_lock.move_and_getimage(expect_robot_pose)
                        if not flag:
                            break
                        flag, objpoint, imgpoint = self.board.getObjImgPointList(rgb_image)
                        if not flag:
                            break
                        camerapose = self.board.extrinsic(imgpoint, objpoint, self.camera.intrinsic, self.camera.dist)
                        self.objpoint_list.append(objpoint)
                        self.imgpoint_list.append(imgpoint)
                        self.Hend2base.append(expect_robot_pose)
                        self.Hobj2camera.append(camerapose)
                        self.rgb_image.append(rgb)
                        self.depth_image.append(depth)
                        flag = self.handeye_cali()
                        if flag:
                            pbar.update(len(self.rgb_image))
                        #print("ias finish:",len(self.image))
                        if len(self.rgb_image) >= 30:
                            pbar.finish()
                            return
            d_min -= 0.05

    

    def run(self,x_qt,y_qt,z_qt):
        # 如果初始化失败
        # if self.init_robot_pose == None:
        #     return "invalid robotpose"
        print("1.init_handeye begin")
        self.init_handeye()
        print("2.sovle init_handeye begin")
        flag=self.handeye_cali()
        if not flag:
            print("init error")
            return False

        # simple_campose为初始相机列表[[4*4],[],...,[]]
        print(1)
        simple_campose = self.camera_pose_simple(1)
        print(2)
        simple_campose = self.select_pose_by_view(simple_campose)

        method_list = {0: "no_Local", 1: "std", 3: 'no_local_std', 5: "random"}
        widgets = [method_list[self.next_step_method], Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA()]
        pbar = ProgressBar(widgets=widgets, maxval=self.picture_number).start()

        pbarstr = "externsic|ProgressBar|正在标定，已完成" + str(round(len(self.rgb_image) / self.picture_number * 100.0, 2)) + "%。"
        self.camera.sendProcessBar(pbarstr)

        #cam_list = simple_campose
        index1=0
        txtrobotpose=[]
        numProcess50 = 0
        while(self.picture_number - len(self.rgb_image)):
            print(3)
            random_simple_pose = simple_campose
            print(4)
            cam_list = self.score_main_multi(random_simple_pose)
            # print('len cam list ',len(cam_list))
            for pose in cam_list:
                #cam_list.pop(0)
                #print("pop(0):",len(cam_list))
                # 利用数据更新迭代
                # robot_pose -- 4*4matrix
                robot_pose = self.get_Expect_robot_pose(pose)
                testpose=robot_pose[:3, 3]
                thres_x=self.init_robot_pose[0][3]
                thres_y=self.init_robot_pose[1][3]
                # print('qtxy',float(x_qt),float(y_qt))
                if(abs(testpose[0] - thres_x) > float(x_qt) or abs(testpose[1] - thres_y) > float(y_qt) ):   #约束x y
                    # print(".....come in constrain xy.....")
                    continue
                # 检查该位置是否可达
                flag = self.robot.moveable(robot_pose,z_qt)
                print(flag)
                if not flag:
                    continue
                # 与机器人通信
                flag, robot_pose, robot_pose_str = self.robot.move_mat(robot_pose)
                if not flag:
                    continue
                print("robot move:",index1)
                # 与相机通信得到图片
                if self.camera.type == '3d':
                    flag, rgb_image, depth_image, img_RootPath = self.camera.get_rgb_depth_image()
                    if flag:
                        cv2.imwrite(str(index1)+".jpg",rgb_image)
                        cv2.imwrite(str(index1) + "depth.jpg", depth_image)
                        txtrobotpose.append(robot_pose_str)
                    else:
                        continue
                    print("get_img3d",index1)
                elif self.camera.type == '2d':
                    flag, rgb_image , img_RootPath= self.camera.get_rgb_image()
                    if flag:
                        cv2.imwrite(str(index1)+".jpg",rgb_image)
                        print("get_img2d",index1)
                    else:
                        continue
                else:
                    pass
                index1=index1+1
                if not flag:
                    data = {'DepthMap': [], 'PointCloudZ': []}
                    data_path, imgName_list, robot_pose = load_data.load_data(img_RootPath, data, imgName_list=[], robot_pose=[])
                    os.remove(imgName_list[-1])
                    os.remove(data_path['PointCloudZ'][-1])
                    continue

                # rgb_image = cv2.undistort(rgb_image, self.camera.intrinsic,self.camera.dist)
                flag, objpoint, imgpoint = self.board.getObjImgPointList(rgb_image)
                # circle
                # flag, objpoint, imgpoint = self.board.getObjImgPointListFromCircle(rgb_image, 5, 4)
                if not flag:
                    print("no see...!!!!!!!!!!!!!!!!!!!!!!!!!")
                    continue
                # for i in range(objpoint.shape[0]):
                #     print((int(imgpoint[i][0]),int(imgpoint[i][1])))
                #     cv2.circle(rgb_image,(int(imgpoint[i][0]),int(imgpoint[i][1])),1,(0,255,0),2)
                #     # cv2.nameWindow('rgb',0)
                #     cv2.imshow('rgb',rgb_image)
                #     print(objpoint[i])
                #     cv2.waitKey(0)
                if self.camera.type=='3d':
                    flag, camerapose, depthcorner, markercorner = getCameraposeFromDepth(self.board, rgb_image, depth_image,self.camera.intrinsic,self.camera.dist)
                    self.depth_image.append(depth_image)
                    self.depth_corner.append(depthcorner)  # size : n x 3
                    self.marker_corner.append(markercorner)  # size : n x 3

                    camerapose = self.board.extrinsic(imgpoint, objpoint, self.camera.intrinsic, self.camera.dist)
                else:
                    camerapose = self.board.extrinsic(imgpoint, objpoint, self.camera.intrinsic, self.camera.dist)
                # 尝试结束
                self.objpoint_list.append(objpoint)
                self.imgpoint_list.append(imgpoint)
                self.Hend2base.append(robot_pose)
                self.Hobj2camera.append(camerapose)
                self.rgb_image.append(rgb_image)
                self.save_txt.append(robot_pose_str)
                break

            flag = self.handeye_cali()
            print("opt cali...")
            if flag:
                if self.error < 0.0007 and len(self.rgb_image) > 9:
                    pbar.update(self.picture_number)
                    pbarstr = "externsic|ProgressBar|正在标定，已完成" + str(100.00) + "%。"
                    self.camera.sendProcessBar(pbarstr)
                    pbar.finish()
                    self.robot.release()
                    print("end!!!")
                    print(self.Hx)
                    return "success"
                print("true")
                pbar.update(len(self.rgb_image))
                pbarstr = "externsic|ProgressBar|正在标定，已完成" + str(round(len(self.rgb_image) / self.picture_number * 100.0,2)) + "%。"
                self.camera.sendProcessBar(pbarstr)
            if(int(len(self.rgb_image) / self.picture_number * 100.0) == 50):
                numProcess50 += 1
                print('numProcess',numProcess50)
            if(numProcess50 > 6):
                str1='msg|'+'Because of the picture or other problems,please use Auto2 calibration or change threshold or move the init pose.'
                self.camera.sendProcessBar(str1)
                assert False,str1
        pbar.finish()

        self.robot.release()
        # self.save_robotpose()
        print("end!!!")
        # print(self.result)
        print(self.Hx)

        # que = transforms3d.quaternions.mat2quat(self.Hx[:3, :3])
        # quet= self.Hx[:3, 3]
        # print("four HX")
        # print(que,quet)
        # print('txtrobotpose', txtrobotpose)
        # print('intr', self.camera.intrinsic)
        # print('uns', self.camera.dist)
        return "success"
    def run2(self,x_qt,y_qt,z_qt):
        # 如果初始化失败
        self.picture_number = 25
        print("1.init_handeye begin")
        self.init_handeye()
        print("2.sovle init_handeye begin")
        flag=self.handeye_cali()
        if not flag:
            print("init error")
            return False

        # simple_campose为初始相机列表[[4*4],[],...,[]]
        print(1)
        simple_campose = self.camera_pose_simple(1)
        print(2)
        simple_campose = self.select_pose_by_view(simple_campose)

        method_list = {0: "no_Local", 1: "std", 3: 'no_local_std', 5: "random"}
        widgets = [method_list[self.next_step_method], Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA()]
        pbar = ProgressBar(widgets=widgets, maxval=self.picture_number).start()

        pbarstr = "externsic|ProgressBar|正在标定，已完成" + str(round(len(self.rgb_image) / self.picture_number * 100.0, 2)) + "%。"
        self.camera.sendProcessBar(pbarstr)

        #cam_list = simple_campose
        index1=0
        txtrobotpose=[]
        # numProcess50 = 0
        while(self.picture_number - len(self.rgb_image)):
            print(3)
            random_simple_pose = simple_campose
            print(4)
            cam_list = self.score_main_multi(random_simple_pose)
            # print('len cam list ',len(cam_list))
            for pose in cam_list:
                #cam_list.pop(0)
                #print("pop(0):",len(cam_list))
                # 利用数据更新迭代
                # robot_pose -- 4*4matrix
                robot_pose = self.get_Expect_robot_pose(pose)
                testpose=robot_pose[:3, 3]
                thres_x=self.init_robot_pose[0][3]
                thres_y=self.init_robot_pose[1][3]
                # print('qtxy',float(x_qt),float(y_qt))
                if(abs(testpose[0] - thres_x) > float(x_qt) or abs(testpose[1] - thres_y) > float(y_qt) ):   #约束x y
                    # print(".....come in constrain xy.....")
                    continue
                # 检查该位置是否可达
                flag = self.robot.moveable(robot_pose,z_qt)
                print(flag)
                if not flag:
                    continue
                # 与机器人通信
                flag, robot_pose, robot_pose_str = self.robot.move_mat(robot_pose)
                if not flag:
                    continue
                print("robot move:",index1)
                # 与相机通信得到图片
                if self.camera.type == '3d':
                    flag, rgb_image, depth_image, img_RootPath = self.camera.get_rgb_depth_image()
                    if flag:
                        cv2.imwrite(str(index1)+".jpg",rgb_image)
                        cv2.imwrite(str(index1) + "depth.jpg", depth_image)
                        txtrobotpose.append(robot_pose_str)
                    else:
                        continue
                    print("get_img3d",index1)
                elif self.camera.type == '2d':
                    flag, rgb_image , img_RootPath= self.camera.get_rgb_image()
                    if flag:
                        cv2.imwrite(str(index1)+".jpg",rgb_image)
                        print("get_img2d",index1)
                    else:
                        continue
                else:
                    pass
                index1=index1+1
                if not flag:
                    data = {'DepthMap': [], 'PointCloudZ': []}
                    data_path, imgName_list, robot_pose = load_data.load_data(img_RootPath, data, imgName_list=[], robot_pose=[])
                    os.remove(imgName_list[-1])
                    os.remove(data_path['PointCloudZ'][-1])
                    continue

                # rgb_image = cv2.undistort(rgb_image, self.camera.intrinsic,self.camera.dist)
                flag, objpoint, imgpoint = self.board.getObjImgPointList(rgb_image)
                # circle
                # flag, objpoint, imgpoint = self.board.getObjImgPointListFromCircle(rgb_image, 5, 4)
                if not flag:
                    print("no see...!!!!!!!!!!!!!!!!!!!!!!!!!")
                    continue
                if self.camera.type=='3d':
                    flag, camerapose, depthcorner, markercorner = getCameraposeFromDepth(self.board, rgb_image, depth_image,self.camera.intrinsic,self.camera.dist)
                    self.depth_image.append(depth_image)
                    self.depth_corner.append(depthcorner)  # size : n x 3
                    self.marker_corner.append(markercorner)  # size : n x 3

                    camerapose = self.board.extrinsic(imgpoint, objpoint, self.camera.intrinsic, self.camera.dist)
                else:
                    camerapose = self.board.extrinsic(imgpoint, objpoint, self.camera.intrinsic, self.camera.dist)
                # 尝试结束
                self.objpoint_list.append(objpoint)
                self.imgpoint_list.append(imgpoint)
                self.Hend2base.append(robot_pose)
                self.Hobj2camera.append(camerapose)
                self.rgb_image.append(rgb_image)
                self.save_txt.append(robot_pose_str)
                break
            if(len(self.rgb_image) < self.picture_number):
                flag = self.handeye_cali2()
            else:
                strwait='externsic|ProgressBar|computing...please wait...'
                self.camera.sendProcessBar(strwait)
                flag = self.semi_handeye_cali()
                print('cali pic number is enough')
            print("opt cali...")
            if flag:
                if self.error < 0.001 and len(self.rgb_image) > 9:
                    pbar.update(self.picture_number)
                    pbarstr = "externsic|ProgressBar|正在标定，已完成" + str(100.00) + "%。"
                    self.camera.sendProcessBar(pbarstr)
                    pbar.finish()
                    self.robot.release()
                    print("end!!!")
                    print(self.Hx)
                    return "success"
                print("true")
                pbar.update(len(self.rgb_image))
                pbarstr = "externsic|ProgressBar|正在标定，已完成" + str(round(len(self.rgb_image) / self.picture_number * 100.0,2)) + "%。"
                self.camera.sendProcessBar(pbarstr)
                # numProcess50 += 1
                # if(numProcess50 > 6):
                #     str1='msg|'+'Because of the picture or other problems,please use Manual calibration or change threshold.'
                #     self.camera.sendProcessBar(str1)
                #     assert False,str1
        pbar.finish()

        self.robot.release()
        # self.save_robotpose()
        print("end!!!")
        print(self.Hx)
        # que = transforms3d.quaternions.mat2quat(self.Hx[:3, :3])
        # quet= self.Hx[:3, 3]
        # print("four HX")
        # print(que,quet)
        # print('txtrobotpose', txtrobotpose)
        # print('intr', self.camera.intrinsic)
        # print('uns', self.camera.dist)
        return "success"
    def semirun(self,x_qt,y_qt,z_qt,posetxtpath):
        # 如果初始化失败
        # if self.init_robot_pose == None:
        #     return "invalid robotpose"
        print("1.semi_init_handeye begin")
        self.semi_init_handeye(posetxtpath)
        print("2.sovle init_handeye begin")
        strwait = 'externsic|ProgressBar|computing...please wait...'
        self.camera.sendProcessBar(strwait)
        flag = self.semi_handeye_cali()
        pbarstr = "externsic|ProgressBar|正在标定，已完成" + str(100.00) + "%。"
        self.camera.sendProcessBar(pbarstr)
        print(pbarstr)
        if not flag:
            print("data error")
            return False
        self.robot.release()
        # self.save_robotpose()
        print("end!!!")
        # print(self.result)
        print(self.Hx)
        return "success"

    def runonline(self):
        # 如果初始化失败
        # if self.init_robot_pose == None:
        #     return "invalid robotpose"
        print("1.init_handeye begin")
        self.init_handeye()
        print("2.sovle init_handeye begin")
        flag=self.handeye_cali()
        if not flag:
            print("init error")
            return False

        # simple_campose为初始相机列表[[4*4],[],...,[]]
        print(1)
        simple_campose = self.camera_pose_simple(1)
        print(2)
        simple_campose = self.select_pose_by_view(simple_campose)

        method_list = {0: "no_Local", 1: "std", 3: 'no_local_std', 5: "random"}
        widgets = [method_list[self.next_step_method], Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA()]
        pbar = ProgressBar(widgets=widgets, maxval=self.picture_number).start()

        #cam_list = simple_campose
        index1=0
        while(self.picture_number - len(self.rgb_image)):
            print(3)
            random_simple_pose = simple_campose
            # print("random_simple_pose size:",len(random_simple_pose))
            # print("random_simple_pose1:",random_simple_pose[0])
            # print("random_simple_pose2:",random_simple_pose[1])
            # print("random_simple_pose3:",random_simple_pose[2])
            #print("random_simple_pose:",random_simple_pose[3])
            #print("random_simple_pose:",random_simple_pose[4])
            print(4)
            cam_list = self.score_main_multi(random_simple_pose)

            # print("cam_list size:",len(cam_list))
            # print("cam_list1:",cam_list[0])
            # print("cam_list2:",cam_list[1])
            # print("cam_list3:",cam_list[2])
            #print("cam_list:",cam_list[3])
            #print("cam_list:",cam_list[4])
            txtrobotpose=[]
            for pose in cam_list:
                #cam_list.pop(0)
                #print("pop(0):",len(cam_list))
                # 利用数据更新迭代
                # robot_pose -- 4*4matrix
                robot_pose = self.get_Expect_robot_pose(pose)

                # print("cam pose")
                # print(pose)
                # print("robot pose")
                # print(robot_pose)
                # print("Hx")
                # print(self.Hx)
                # print("Hy")
                # print(self.Hy)
                # print("...........................................................................")
                # print("cam pose1")
                # print(self.Hobj2camera[0])
                # print("robot pose1")
                # print(self.Hend2base[0])
                # print("Hx")
                # print(self.Hx)
                # print("Hy")
                # print(self.Hy)

                testpose=robot_pose[:3, 3]
                thres_x=self.init_robot_pose[0][3]
                thres_y=self.init_robot_pose[1][3]
                if(abs(testpose[0]-thres_x)>0.3 or abs(testpose[1]-thres_y)>0.3):   #约束x y
                    print(".....come in constrain xy.....")
                    continue

                # 检查该位置是否可达
                flag = self.robot.moveable(robot_pose)
                #if(abs(self.Hend2base[-1][0][3]-robot_pose[0][3])<0.2 and abs(self.Hend2base[-1][1][3]-robot_pose[1][3])<0.2):
                #    continue
                print(flag)
                if not flag:
                    #print("continue")
                    continue

                # 与机器人通信
                flag, robot_pose, robot_pose_str = self.robot.move_mat(robot_pose)
                if not flag:
                    continue
                print("robot move:",index1)
                # 与相机通信得到图片
                if self.camera.type == '3d':
                    flag, rgb_image, depth_image, img_RootPath = self.camera.get_rgb_depth_image()
                    if flag:
                        cv2.imwrite(str(index1)+".jpg",rgb_image)
                        cv2.imwrite(str(index1) + "depth.jpg", depth_image)
                        txtrobotpose.append(robot_pose_str)
                    print("get_img3d",index1)
                elif self.camera.type == '2d':
                    flag, rgb_image, img_RootPath = self.camera.get_rgb_image()
                    cv2.imwrite(str(index1)+".jpg",rgb_image)
                    print("get_img2d",index1)
                else:
                    pass
                index1=index1+1
                if not flag:
                    data = {'DepthMap': [], 'PointCloudZ': []}
                    data_path, imgName_list, robot_pose = load_data.load_data(img_RootPath, data, imgName_list=[], robot_pose=[])
                    os.remove(imgName_list[-1])
                    os.remove(data_path['PointCloudZ'][-1])
                    continue

                flag, objpoint, imgpoint = self.board.getObjImgPointList(rgb_image)
                if not flag:
                    print("no see...")
                    continue
                # 计算机器人位姿下精确的外参
                cameraposepnp = self.board.extrinsic(imgpoint, objpoint, self.camera.intrinsic, self.camera.dist)
                # 尝试直接用深度图
                flag, camerapose, depthcorner,markercorner = getCameraposeFromDepth(self.board, rgb_image, depth_image, self.camera.intrinsic,self.camera.dist)
                if (flag):
                    print("svd RT:", camerapose)
                    print("pnp RT:",cameraposepnp)
                #     print("depthcorner:",depthcorner)
                #     print("markercorner:",markercorner)
                # 尝试结束
                self.objpoint_list.append(objpoint)
                self.imgpoint_list.append(imgpoint)
                self.Hend2base.append(robot_pose)
                self.Hobj2camera.append(cameraposepnp)
                self.rgb_image.append(rgb_image)
                self.depth_image.append(depth_image)
                self.save_txt.append(robot_pose_str)
                self.depth_corner.append(depthcorner)  #size : n x 3
                self.marker_corner.append(markercorner)#size : n x 3
                break

            flag = self.handeye_cali()
            print("opt cali...")
            if flag:
                print("true")
                pbar.update(len(self.rgb_image))
        pbar.finish()
        self.robot.release()
        self.save_robotpose()
        print("end!!!")
        print(self.result)
        print(self.Hx)

        que = transforms3d.quaternions.mat2quat(self.Hx[:3, :3])
        quet= self.Hx[:3, 3]
        print("four HX")
        print(que,quet)
        print('txtrobotpose',txtrobotpose)
        print('intr',self.camera.intrinsic)
        print('uns',self.camera.dist)

        return "success"
    #www
    def get3dPointfrom2dcam(self,Hx,f):
        print(self.camera.intrinsic)
        print(self.camera.dist)
        if f == 1:
            print("cali")
            flag, rgb_image, depth_image, img_RootPath = self.camera.get_rgb_depth_image()
            # flag, rgb_image, img_RootPath = self.camera.get_rgb_image()
            # rgb_image=cv2.imread("6.jpg")
            cv2.imwrite("camera2d.jpg", rgb_image)
            undistortimg = cv2.undistort(rgb_image, self.camera.intrinsic, self.camera.dist)
            undistortdep_img = cv2.undistort(depth_image, self.camera.intrinsic, self.camera.dist)

            flag, objpoint, imgpoint = self.board.getObjImgPointList(undistortimg)
            if not flag:
                print("no see...")
                return False,None
            print(imgpoint[0])
            print(undistortdep_img.shape)   #(1544, 2064, 3)   行数、列数
            print(undistortimg.shape)
            z = undistortdep_img[int(round(imgpoint[0][1])),int(round(imgpoint[0][0]))]
            fx, fy = self.camera.intrinsic[0, 0], self.camera.intrinsic[1, 1]
            cx, cy = self.camera.intrinsic[0, 2], self.camera.intrinsic[1, 2]
            z = z / 1000
            print("depth",z)
            x = (imgpoint[0][0] - cx) * z / fx
            y = (imgpoint[0][1] - cy) * z / fy
            xyz=np.array([x,y,z,1])
            robotcoor3d = np.dot(Hx,xyz)
            robotcoor3d = robotcoor3d[:]/robotcoor3d[3]
            # 计算机器人位姿下精确的外参
            cameraposepnp = self.board.extrinsic(imgpoint, objpoint, self.camera.intrinsic, self.camera.dist)
            flag, camerapose, depthcorner, markercorner = getCameraposeFromDepth(self.board, rgb_image, depth_image,self.camera.intrinsic, self.camera.dist)
            M, mask = cv2.findHomography(imgpoint, objpoint, cv2.RANSAC, 5.0)
            print("m",M)
            cv2.circle(undistortimg,(int(imgpoint[0][0]),int(imgpoint[0][1])),2,(0,0,255))
            #cv2.imshow("1",undistortimg)
            #cv2.waitKey(0)
            imgpoint=np.append(imgpoint,np.ones([imgpoint.shape[0],1]),1)
            objpoint=np.append(objpoint,np.ones([objpoint.shape[0],1]),1)
            #print('objpoint',objpoint)
            imgfromojb=np.dot(np.linalg.inv(M),objpoint[0])
            imgfromojb = imgfromojb[:] / imgfromojb[2]
            print('img',imgpoint[0])
            print("imgformobj",imgfromojb)
            cv2.circle(undistortimg, (int(imgfromojb[0]),int(imgfromojb[1])), 10, (0, 255, 0))
            cv2.imwrite("camera2Dcircle.jpg", undistortimg)
            obj=np.dot(M,imgpoint[0])
            obj=obj[:]/obj[2]
            print('marker', objpoint[0])
            print('H_marker',obj)

            obj1=np.array([obj[0],obj[1],0,1])
            robotcoor=np.dot(Hx,np.dot(cameraposepnp,obj1))
            robotcoor1 = np.dot(Hx, np.dot(camerapose, obj1))
            robotcoor1 = robotcoor1[:] / robotcoor1[3]
            #print(obj1,obj1)
            print('camerapose pnp',cameraposepnp)
            print('camerapose svd',camerapose)
            print('Hx',Hx)
            print('robotcoor',robotcoor)
            robotcoor = robotcoor[:] / robotcoor[3]
            print('robotcoor_H_PNP',robotcoor)
            print('robotcoor_H_SVD',robotcoor1)
            print('robotcoor3d',robotcoor3d)
        elif f == 0:
            print('test')
            flag, rgb_image, depth_image, img_RootPath = self.camera.get_rgb_depth_image()
            # flag, rgb_image, img_RootPath = self.camera.get_rgb_image()
            # rgb_image=cv2.imread("6.jpg")
            cv2.imwrite("camera2d.jpg", rgb_image)
            undistortimg = cv2.undistort(rgb_image, self.camera.intrinsic, self.camera.dist)
            undistortdep_img = cv2.undistort(depth_image, self.camera.intrinsic, self.camera.dist)

            flag, objpoint, imgpoint = self.board.getObjImgPointList(undistortimg)
            if not flag:
                print("no see...")
                return False, None
            print('imgpoint[0]',imgpoint[0])

            #hn test
            imgpoint[0][0]=98
            imgpoint[0][1]=491
            #hn test

            #print(undistortdep_img.shape)  # (1544, 2064, 3)   行数、列数
            #print(undistortimg.shape)
            z = undistortdep_img[int(round(imgpoint[0][1])), int(round(imgpoint[0][0]))]
            fx, fy = self.camera.intrinsic[0, 0], self.camera.intrinsic[1, 1]
            cx, cy = self.camera.intrinsic[0, 2], self.camera.intrinsic[1, 2]
            z = z / 1000
            print("depth", z)
            x = (imgpoint[0][0] - cx) * z / fx
            y = (imgpoint[0][1] - cy) * z / fy
            xyz = np.array([x, y, z, 1])
            print("xyz",xyz)
            robotcoor3d = np.dot(Hx, xyz)
            robotcoor3d = robotcoor3d[:] / robotcoor3d[3]
            # 计算机器人位姿下精确的外参
            cameraposepnp = np.array([[-0.99295746, 0.02499976, 0.11580371, -0.05779318],
                                      [-0.03066471, -0.99840528, -0.04739794, -0.17620566],
                                      [0.1144341, -0.05061522, 0.99214058, 2.33887924],
                                      [0., 0., 0., 1.]])
            flag, camerapose, depthcorner, markercorner = getCameraposeFromDepth(self.board, rgb_image, depth_image,self.camera.intrinsic,self.camera.dist)
            M = np.array([[-1.09649540e-03 ,-2.69370193e-05 , 1.12534039e+00],
                        [ 2.21430490e-05 ,-1.08416943e-03  ,6.57472712e-01],
                        [ 5.43363609e-05 ,-1.18683153e-05  ,1.00000000e+00]])
            print("m", M)
            cv2.circle(undistortimg, (int(imgpoint[0][0]), int(imgpoint[0][1])), 2, (0, 0, 255) , 2)
            # cv2.imshow("1",undistortimg)
            # cv2.waitKey(0)
            imgpoint = np.append(imgpoint, np.ones([imgpoint.shape[0], 1]), 1)
            objpoint = np.append(objpoint, np.ones([objpoint.shape[0], 1]), 1)
            # print('objpoint',objpoint)
            imgfromojb = np.dot(np.linalg.inv(M), objpoint[0])
            imgfromojb = imgfromojb[:] / imgfromojb[2]
            print('img', imgpoint[0])
            print("imgformobj", imgfromojb)
            obj = np.dot(M, imgpoint[0])
            obj = obj[:] / obj[2]
            print('marker', objpoint[0])
            print('H_marker', obj)

            obj1 = np.array([obj[0], obj[1], 0, 1])
            recamera = np.dot(cameraposepnp, obj1)
            recamera = recamera[:] / recamera[3]
            rex=fx*recamera[0]/recamera[2]+cx
            rey=fy*recamera[1]/recamera[2]+cy
            cv2.circle(undistortimg, (int(rex), int(rey)), 2, (255, 0, 0),2)
            cv2.imwrite("camera2Dcircle.jpg", undistortimg)
            robotcoor = np.dot(Hx, np.dot(cameraposepnp, obj1))
            robotcoor1 = np.dot(Hx, np.dot(camerapose, obj1))
            robotcoor1 = robotcoor1[:] / robotcoor1[3]
            # print(obj1,obj1)
            print('camerapose pnp', cameraposepnp)
            print('camerapose svd', camerapose)
            print('Hx', Hx)
            print('robotcoor', robotcoor)
            robotcoor = robotcoor[:] / robotcoor[3]
            print('robotcoor_H_PNP', robotcoor)
            print('robotcoor_H_SVD', robotcoor1)
            print('robotcoor3d', robotcoor3d)
        elif f==2:
            print("cali--line")
            # flag, rgb_image, depth_image, img_RootPath = self.camera.get_rgb_depth_image()
            # flag, rgb_image, img_RootPath = self.camera.get_rgb_image()
            # rgb_image=cv2.imread("6.jpg")
            # cv2.imwrite("camera2d.jpg", rgb_image)
            # undistortimg = cv2.undistort(rgb_image, self.camera.intrinsic, self.camera.dist)
            # undistortdep_img = cv2.undistort(depth_image, self.camera.intrinsic, self.camera.dist)
            undistortimg = cv2.imread('/home/speedbot/Documents/speedEye/qt_calibration/auto_calibration1/2dtest/new1/RGB/image2.bmp')
            undistortimg2 = cv2.imread('/home/speedbot/Documents/speedEye/qt_calibration/auto_calibration1/2dtest/new1/RGB/rgb_1611645093.png')
            flag, objpoint, imgpoint = self.board.getObjImgPointList(undistortimg,1)
            if not flag:
                print("no see...")
                return False, None

            # 计算机器人位姿下精确的外参
            flag, objpoint1, imgpoint1 = self.board.getObjImgPointList(undistortimg2)
            cameraposepnp = self.board.extrinsic(imgpoint1, objpoint1, self.camera.intrinsic, self.camera.dist)

            M, mask = cv2.findHomography(imgpoint, objpoint, cv2.RANSAC, 5.0)
            print("m", M)
            cv2.circle(undistortimg, (int(imgpoint[0][0]), int(imgpoint[0][1])), 2, (0, 0, 255))
            # cv2.imshow("1",undistortimg)
            # cv2.waitKey(0)
            imgpoint = np.append(imgpoint, np.ones([imgpoint.shape[0], 1]), 1)
            objpoint = np.append(objpoint, np.ones([objpoint.shape[0], 1]), 1)
            # print('objpoint',objpoint)
            imgfromojb = np.dot(np.linalg.inv(M), objpoint[0])
            imgfromojb = imgfromojb[:] / imgfromojb[2]
            print('img', imgpoint[0])
            print("imgformobj", imgfromojb)
            cv2.circle(undistortimg, (int(imgfromojb[0]), int(imgfromojb[1])), 10, (0, 255, 0))
            cv2.imwrite("camera2Dcircle.jpg", undistortimg)
            obj = np.dot(M, imgpoint[0])
            obj = obj[:] / obj[2]
            print('marker', objpoint[0])
            print('H_marker', obj)

            obj1 = np.array([obj[0], obj[1], 0, 1])
            robotcoor = np.dot(Hx, np.dot(cameraposepnp, obj1))
            # print(obj1,obj1)
            print('camerapose pnp', cameraposepnp)
            print('Hx', Hx)
            print('robotcoor', robotcoor)
            robotcoor = robotcoor[:] / robotcoor[3]
            print('robotcoor_H_PNP', robotcoor)
        elif f==3:
            print('test')

            # flag, rgb_image, depth_image, img_RootPath = self.camera.get_rgb_depth_image()
            # # flag, rgb_image, img_RootPath = self.camera.get_rgb_image()
            # # rgb_image=cv2.imread("6.jpg")
            # cv2.imwrite("camera2d.jpg", rgb_image)
            # undistortimg = cv2.undistort(rgb_image, self.camera.intrinsic, self.camera.dist)
            # undistortdep_img = cv2.undistort(depth_image, self.camera.intrinsic, self.camera.dist)
            undistortimg = cv2.imread('/home/speedbot/Documents/speedEye/qt_calibration/auto_calibration1/2dtest/new1/RGB/image4.bmp')
            flag, objpoint, imgpoint = self.board.getObjImgPointList(undistortimg)
            if not flag:
                print("no see...")
                return False, None
            print('imgpoint[0]', imgpoint[0])

            # hn test
            imgpoint[0][0] = 98
            imgpoint[0][1] = 491
            # hn test


            fx, fy = self.camera.intrinsic[0, 0], self.camera.intrinsic[1, 1]
            cx, cy = self.camera.intrinsic[0, 2], self.camera.intrinsic[1, 2]

            # 计算机器人位姿下精确的外参
            cameraposepnp = np.array([[-0.99295746, 0.02499976, 0.11580371, -0.05779318],
                                      [-0.03066471, -0.99840528, -0.04739794, -0.17620566],
                                      [0.1144341, -0.05061522, 0.99214058, 2.33887924],
                                      [0., 0., 0., 1.]])

            M = np.array([[-1.09649540e-03, -2.69370193e-05, 1.12534039e+00],
                          [2.21430490e-05, -1.08416943e-03, 6.57472712e-01],
                          [5.43363609e-05, -1.18683153e-05, 1.00000000e+00]])
            print("m", M)
            cv2.circle(undistortimg, (int(imgpoint[0][0]), int(imgpoint[0][1])), 2, (0, 0, 255), 2)
            # cv2.imshow("1",undistortimg)
            # cv2.waitKey(0)
            imgpoint = np.append(imgpoint, np.ones([imgpoint.shape[0], 1]), 1)
            objpoint = np.append(objpoint, np.ones([objpoint.shape[0], 1]), 1)
            # print('objpoint',objpoint)
            imgfromojb = np.dot(np.linalg.inv(M), objpoint[0])
            imgfromojb = imgfromojb[:] / imgfromojb[2]
            print('img', imgpoint[0])
            print("imgformobj", imgfromojb)
            obj = np.dot(M, imgpoint[0])
            obj = obj[:] / obj[2]
            print('marker', objpoint[0])
            print('H_marker', obj)

            obj1 = np.array([obj[0], obj[1], 0, 1])
            recamera = np.dot(cameraposepnp, obj1)
            recamera = recamera[:] / recamera[3]
            rex = fx * recamera[0] / recamera[2] + cx
            rey = fy * recamera[1] / recamera[2] + cy
            cv2.circle(undistortimg, (int(rex), int(rey)), 2, (255, 0, 0), 2)
            cv2.imwrite("camera2Dcircle.jpg", undistortimg)
            robotcoor = np.dot(Hx, np.dot(cameraposepnp, obj1))
            # print(obj1,obj1)
            print('camerapose pnp', cameraposepnp)
            print('Hx', Hx)
            print('robotcoor', robotcoor)
            robotcoor = robotcoor[:] / robotcoor[3]
            print('robotcoor_H_PNP', robotcoor)

    def save_robotpose(self):
        with open(self.camera.img_RootPath + '/robotpose.txt', 'a') as wf:
            for pose in self.save_txt:
                wf.write(pose)
                wf.write('\n')

    # def save_result(self, file):
    #     from auto import utils
    #     utils.json_save(self.result, file)

    def set_select_method(self, method_id):
        self.next_step_method = method_id