# -*- coding:utf-8 -*-
import numpy as np
from AprilTag.board import apriltagBoard
import sys
ros_cv2_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_cv2_path in sys.path:
    sys.path.remove(ros_cv2_path)
    import cv2
    sys.path.append(ros_cv2_path)
else:
    import cv2

def getCameraMatrix(tags_list, board, img_size):
    obj_points_list = []
    img_conner_list = []
    for tags in tags_list:
        n = len(tags)
        if n == 0:
            continue
        # 标定板坐标系
        a = np.ndarray([5 * n, 1, 3], dtype=np.float32)
        # 像素坐标系
        b = np.ndarray([5 * n, 1, 2], dtype=np.float32)
        for i in range(n):
            tag = tags[i]
            center, conner = board.getPointsbyTagId(tag.tag_id)
            a[5 * i, 0, :2] = center[:]
            a[5 * i + 1, 0, :2] = conner[0, :]
            a[5 * i + 2, 0, :2] = conner[1, :]
            a[5 * i + 3, 0, :2] = conner[2, :]
            a[5 * i + 4, 0, :2] = conner[3, :]
            a[5 * i, 0, 2] = 0
            a[5 * i + 1, 0, 2] = 0
            a[5 * i + 2, 0, 2] = 0
            a[5 * i + 3, 0, 2] = 0
            a[5 * i + 4, 0, 2] = 0
            b[5 * i, 0, :2] = tag.center[:]
            b[5 * i + 1, 0, :] = tag.corners[0, :]
            b[5 * i + 2, 0, :] = tag.corners[1, :]
            b[5 * i + 3, 0, :] = tag.corners[2, :]
            b[5 * i + 4, 0, :] = tag.corners[3, :]

        obj_points_list.append(a)
        img_conner_list.append(b)
        # ret, mtx, dist, rvecs, tvec = cv2.calibrateCamera([a], [b], img_size, None, None)
        # mtx1.append(mtx)
    ret, mtx, dist, rvecs, tvec = cv2.calibrateCamera(obj_points_list, img_conner_list, img_size, None, None)
    return ret, mtx, dist

def getCameraMatrix2(objPoint_list,imgPoint_list,img_size):
    obj_points_list = []
    img_conner_list = []

    for i in range(len(objPoint_list)):
        n = objPoint_list[i].shape[0]
        a = np.ndarray([n, 1, 3], dtype=np.float32)
        b = np.ndarray([n, 1, 2], dtype=np.float32)
        a[:, 0, :2] = objPoint_list[i][:, :2]
        a[:, 0, 2] = 0
        b[:, 0, :2] = imgPoint_list[i][:, :2]
        obj_points_list.append(a)
        img_conner_list.append(b)
    ret, mtx, dist, rvecs, tvec = cv2.calibrateCamera(obj_points_list, img_conner_list, img_size, None, None)
    return mtx, dist



