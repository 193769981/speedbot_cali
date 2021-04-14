# -*- coding:utf-8 -*-
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
from cali_refine.refineCamera import refinall_all_param
from cali_refine.refine_extrinsic import refine_extrinsic_lm
# from AprilTag.refine_intrinsic import refinall_intrinsic
#
from AprilTag.cameraMatrix import getCameraMatrix
import os
import platform

if platform.system().lower() == 'linux':
    import dt_apriltags as apriltag
else:
    import pupil_apriltags as apriltag
FIX_CAMERA_MATRIX = 1
UNFIX_CAMERA_MATRIX = 2


def camera_cali(board, imgsize, img_list, verbose=0):
    '''
    角点检测+相机内参标定
    :param board: apriltagBoard对象
    :param imgsize: 图片尺寸
    :param img_list: 图片列表(用于求内参)
    :param verbose: 可视化flag
    :return:
    '''
    tags_list = []
    corners_pixel_list = []
    for img in img_list:
        # 检测角点
        tags = detectTags_img(board, img)
        tags_list.append(tags)
        corners = [tag.corners for tag in tags]
        corners_pixel_list.append(corners)
        if verbose == 1:
            # board对应的角点，以及对应角点在图片上的坐标
            objPoint, imgPoint = getObjImgPointList(tags, board)
            for i in range(imgPoint.shape[0]):
                img = cv2.putText(img, str(i), (int(imgPoint[i, 0]), int(imgPoint[i, 1])), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                  (0, 255, 0), 3)
            cv2.namedWindow("apriltag", cv2.WINDOW_NORMAL)
            cv2.imshow("apriltag", img)
            cv2.waitKey(0)
    assert len(tags_list) != 0, "tagId可能选择错误，请检查！"
    # 得到相机内参，畸变系数
    ret, camera_matrix, discoff = getCameraMatrix(tags_list, board, imgsize)
    return ret, tags_list, corners_pixel_list, camera_matrix, discoff


def camera_cali2(board, imgsize, img_list, verbose=0):
    '''
    相机内参标定，给定文件夹，按照一定的规则排列，就可以得到相应的内参
    :param board:
    :param imgsize:
    :param img_list:
    :param verbose:
    :return:
    '''
    tags_list = []
    for img in img_list:
        tags = detectTags_img(board, img)
        tags_list.append(tags)
        if verbose == 1:
            objPoint, imgPoint = getObjImgPointList(tags, board)
            for i in range(imgPoint.shape[0]):
                img = cv2.putText(img, str(i), (int(imgPoint[i, 0]), int(imgPoint[i, 1])), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                  (0, 255, 0), 3)
            cv2.namedWindow("apriltag", cv2.WINDOW_NORMAL)
            cv2.imshow("apriltag", img)
            cv2.waitKey(0)
    camera_matrix, discoff = getCameraMatrix(tags_list, board, imgsize)
    return tags_list, camera_matrix, discoff


def camera_pose_cali(board, number, root_dir, suffix, camera_matrix, discoff, flag):
    tags_list = []
    camera_pose_list = []
    for i in range(number):
        imgPath = os.path.join(root_dir, str(i) + suffix)
        tags = detectTags(board, imgPath, camera_matrix, verbose=1)
        tags.sort(key=lambda x: x.tag_id)
        camera_pose = getCameraPose(tags, board, camera_matrix)
        camera_pose_list.append(camera_pose)
        tags_list.append(tags)
    if flag == FIX_CAMERA_MATRIX:
        camera_pose_list1 = refine_extrinsic(tags_list, board, camera_matrix, discoff,
                                             camera_pose_list)
        return tags_list, camera_matrix, discoff, camera_pose_list1
    elif flag == UNFIX_CAMERA_MATRIX:
        camera_matrix1, discoff1, camera_pose_list1 = refine_all(tags_list, board, camera_matrix, discoff,
                                                                 camera_pose_list)
        tags_list = []
        camera_pose_list = []
        for i in range(number):
            imgPath = os.path.join(root_dir, str(i) + suffix)
            tags = detectTags(board, imgPath, camera_matrix1)
            camera_pose = getCameraPose(tags, board, camera_matrix1)
            camera_pose_list.append(camera_pose)
            tags_list.append(tags)
        camera_pose_list1 = refine_extrinsic(tags_list, board, camera_matrix1, discoff1,
                                             camera_pose_list)
        return tags_list, camera_matrix1, discoff1, camera_pose_list1


def detectTags(board, imgPath, cameraMatrix=None, verbose=0):
    '''
    检测img中的apriltag，返回一组tag,如果不输入cameraMatrix，tags中不含位姿信息
    :param board: apiriltag.board 包含board的一些参数
    :param img: 需要检测的图片路径
    :param cameraMatrix:相机内参
    :param verbose:可视化
    :return: tags 检测到的tags
    '''
    img = cv2.imread(imgPath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # for windows
    if cameraMatrix is None:
        tags = board.at_detector.detect(gray)
    else:
        camera_param = [cameraMatrix[0, 0], cameraMatrix[1, 1], cameraMatrix[0, 2], cameraMatrix[1, 2]]
        tags = board.at_detector.detect(gray, True, camera_param, board.tag_size)
        if verbose == 1:
            img = drawTagAxis(img, tags, cameraMatrix)
            cv2.namedWindow("apriltag", cv2.WINDOW_NORMAL)
            cv2.imshow("apriltag", img)
            cv2.waitKey(0)
    return tags


def detectTags_img(board, img, cameraMatrix=None, verbose=0):
    '''
    检测img中的apriltag，返回一组tag, 如果不输入cameraMatrix，tags中不含位姿信息(不画坐标轴)
    :param board: apiriltag.board
    包含board的一些参数
    :param img: 需要检测的图片路径
    :param cameraMatrix: 相机内参
    :param verbose: 是否可视化
    :return: tags 检测到的tags
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if cameraMatrix is None:
        try:
            tags = board.at_detector.detect(gray)
        except Exception:
            tags = []
    else:
        camera_param = [cameraMatrix[0, 0], cameraMatrix[1, 1], cameraMatrix[0, 2], cameraMatrix[1, 2]]
        try:
            tags = board.at_detector.detect(gray, True, camera_param, board.tag_size)
        except Exception:
            tags = []
        if verbose == 1:
            img = drawTagAxis(img, tags, cameraMatrix)
            cv2.namedWindow("apriltag", cv2.WINDOW_NORMAL)
            cv2.imshow("apriltag", img)
            cv2.waitKey(0)
    return tags


def getCameraPose(tags, board, cameraMatrix):
    '''
    得到相机的姿态,主要是将每个tag中估计相机姿态使用4分位法进行筛选（主要是用于rgb图像上）
    :param tags: 检测图片的tags
    :param board: apriltag的规格参数
    :param cameraMatrix: 相机内参
    :return: 4*4 旋转矩阵 表示相机的姿态
    '''
    q = np.array([])
    t = np.array([])
    if len(tags) == 0:
        return False, 0
    for tag in tags:
        q = np.append(q, transforms3d.quaternions.mat2quat(tag.pose_R))
        H = np.append(np.append(tag.pose_R, tag.pose_t, 1), np.array([[0, 0, 0, 1]]), 0)
        x, y = np.where(board.tag_id_order == tag.tag_id)
        board_center = board.boardcenter[x[0] * board.marker_X + y[0], :]
        point = np.array([[-board_center[0], -board_center[1], 0, 1]]).T
        # tag中心点在相机坐标系下的齐次坐标
        proj = np.dot(H, point)
        orgin = proj / proj[3, 0]
        # tag中心点在相机坐标系下的三维坐标，都放在t中
        t = np.append(t, orgin[:3, 0])
    q = np.reshape(q, [-1, 4])
    t = np.reshape(t, [-1, 3])
    # 使用四分位法去除异常值
    q1 = q[:, 0]
    q2 = q[:, 1]
    q3 = q[:, 2]
    q4 = q[:, 3]
    Q11 = np.percentile(q1, 25)
    Q31 = np.percentile(q1, 75)
    Q12 = np.percentile(q2, 25)
    Q32 = np.percentile(q2, 75)
    Q13 = np.percentile(q3, 25)
    Q33 = np.percentile(q3, 75)
    Q14 = np.percentile(q4, 25)
    Q34 = np.percentile(q4, 75)
    IQR1 = 1.5 * (Q31 - Q11)
    IQR2 = 1.5 * (Q32 - Q12)
    IQR3 = 1.5 * (Q33 - Q13)
    IQR4 = 1.5 * (Q34 - Q14)
    for i in range(q.shape[0] - 1, -1, -1):
        flag1 = (q[i, 0] < Q11 - IQR1) | (q[i, 0] > Q31 + IQR1)
        flag2 = (q[i, 1] < Q12 - IQR2) | (q[i, 1] > Q32 + IQR2)
        flag3 = (q[i, 2] < Q13 - IQR3) | (q[i, 2] > Q33 + IQR3)
        flag4 = (q[i, 3] < Q14 - IQR4) | (q[i, 3] > Q34 + IQR4)
        if flag1 | flag2 | flag3 | flag4:
            q = np.delete(q, i, axis=0)
    mean_t = np.array([np.mean(t, axis=0)])
    std_t = np.array([np.std(t, axis=0)])
    error_t = np.empty([t.shape[0], 3], dtype=bool)
    error_t[:, :] = np.abs(t[:, :] - mean_t[0, :]) > 3 * std_t[0, :]
    index, _ = np.where(error_t)
    index_t = np.array(list(set(list(index))))
    if len(index_t > 0):
        t = np.delete(t, index_t, axis=0)
    mean_q = np.mean(q, axis=0)
    mean_t = np.mean(t, axis=0)
    pose_R = transforms3d.quaternions.quat2mat(mean_q)
    pose = np.append(np.append(pose_R, np.array([mean_t]).T, 1), np.array([[0, 0, 0, 1]]), 0)
    return True, pose


def getObjImgPointList(tags, board):
    '''
    提取tags中的角点坐标，以及对应的board的坐标
    :param tags: apriltag检测的tag
    :param board: apriltag板，内含apriltag的一些参数
    :return: objPoint board对应角点
            imgPoint 对应角点在图片上的坐标
    '''
    objpoint = np.array([])
    imgpoint = np.array([])
    for tag in tags:
        center, conners = board.getPointsbyTagId(tag.tag_id)
        objpoint = np.append(objpoint, center)
        imgpoint = np.append(imgpoint, tag.center)
        objpoint = np.append(objpoint, conners)
        imgpoint = np.append(imgpoint, tag.corners)
    objpoint = np.reshape(objpoint, [-1, 2]).astype(np.float32)
    imgpoint = np.reshape(imgpoint, [-1, 2]).astype(np.float32)
    return objpoint, imgpoint


# def getCameraPosePnP(tags,board,cameraMatrix,discoff):
#     '''
#     通过pnp方法得到相机的姿态
#     :param tags: 检测图片的tags
#     :param board: apriltag的规格参数
#     :param discoff:相机的畸变参数
#     :param cameraMatrix: 相机内参
#     :return: 4*4 旋转矩阵 表示相机的姿态
#     '''
#     objpoint,imgpoint = getObjImgPointList(tags,board)
#     n = objpoint.shape[0]
#     objpoint = np.append(objpoint,np.zeros([n,1]),1)
#     retval, rvec, tvec,_= cv2.solvePnPRansac(objpoint, imgpoint, cameraMatrix, discoff,reprojectionError=0.01)
#     R = cv2.Rodrigues(rvec)[0]
#     camerapose = np.append(np.append(R, tvec, 1), np.array([[0, 0, 0, 1]]), 0)
#     return camerapose

def refine_all(tags_list, board, cameramatrix, discoff, camerapose_list):
    '''
    优化相机参数，方法，使用lm非线性优化方法
    :param tags_list: list<list<tag>> 所有图片的tag
    :param board: apriltag 板
    :param cameramatrix: 相机内参
    :param discoff: 畸变参数
    :param camerapose_list: 相机姿态（外参列表）
    :return:
    '''
    objpoint_list = []
    imgpoint_list = []
    for tags in tags_list:
        objpoint, imgpoint = getObjImgPointList(tags, board)
        objpoint_list.append(objpoint)
        imgpoint_list.append(imgpoint)
    cameramatrix_refined, discoff_refined, camerapose_list_refined = refinall_all_param(cameramatrix, discoff,
                                                                                        camerapose_list, objpoint_list,
                                                                                        imgpoint_list)
    return cameramatrix_refined, discoff_refined, camerapose_list_refined


def refine_extrinsic(tags_list, board, cameramatrix, discoff, camerapose_list):
    '''
    优化外参
    :param tags_list:
    :param board:
    :param cameramatrix:
    :param discoff:
    :param camerapose_list:
    :return:
    '''
    objpoint_list = []
    imgpoint_list = []
    for tags in tags_list:
        objpoint, imgpoint = getObjImgPointList(tags, board)
        objpoint_list.append(objpoint)
        imgpoint_list.append(imgpoint)
    camerapose_list_refined = refine_extrinsic_lm(cameramatrix, discoff,
                                                  camerapose_list, objpoint_list,
                                                  imgpoint_list)
    return camerapose_list_refined


# def refine_intrinsic(tags_list,board,cameramatrix,discoff,camerapose_list):
#     objpoint_list = []
#     imgpoint_list = []
#     for tags in tags_list:
#         objpoint, imgpoint = getObjImgPointList(tags, board)
#         objpoint_list.append(objpoint)
#         imgpoint_list.append(imgpoint)
#     A,discoff = refinall_intrinsic(cameramatrix, np.array([discoff]),
#                                                  camerapose_list, objpoint_list,
#                                                  imgpoint_list)
#     return A,discoff

# 在每个tag上画出坐标轴
def drawTagAxis(img, tags, cameraMatrix, length=0.015, line_width=2):
    '''
    在图像上画出每个tag的坐标轴，蓝色表示x轴，绿色表示y轴，红色表示z轴
    :param img: 图片
    :param tags:
    :param cameraMatrix: 相机内参
    :param length: 长度，指实际长度
    :return: img：图片
    '''
    point_x = np.array([[length, 0, 0, 1]]).T
    point_y = np.array([[0, length, 0, 1]]).T
    point_z = np.array([[0, 0, length, 1]]).T
    cameraMatrix = np.append(cameraMatrix, np.zeros([3, 1]), 1)
    for tag in tags:
        R = tag.pose_R
        T = tag.pose_t
        H = np.append(np.append(R, T, 1), np.array([[0, 0, 0, 1]]), 0)
        pro_x = np.dot(cameraMatrix, np.dot(H, point_x))
        pro_x = pro_x / pro_x[2, 0]
        pro_y = np.dot(cameraMatrix, np.dot(H, point_y))
        pro_y = pro_y / pro_y[2, 0]
        pro_z = np.dot(cameraMatrix, np.dot(H, point_z))
        pro_z = pro_z / pro_z[2, 0]

        cv2.putText(img, str(tag.tag_id), (int(tag.center[0]) - 30, int(tag.center[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 252, 51), 3)
        img = cv2.line(img, (int(tag.center[0]), int(tag.center[1])), (int(pro_x[0, 0]), int(pro_x[1, 0])), (255, 0, 0),
                       thickness=line_width)
        img = cv2.line(img, (int(tag.center[0]), int(tag.center[1])), (int(pro_y[0, 0]), int(pro_y[1, 0])), (0, 255, 0),
                       thickness=line_width)
        img = cv2.line(img, (int(tag.center[0]), int(tag.center[1])), (int(pro_z[0, 0]), int(pro_z[1, 0])), (0, 0, 255),
                       thickness=line_width)
    return img


# 在图片上画坐标轴
def drawBoardPose(img, camerapose, cameraMatrix, length=0.015, line_width=3):
    '''
    在图片上画出坐标轴
    :param img: 图片
    :param camerapose:相机姿态
    :param cameraMatrix: 相机内参
    :param length: 实际长度
    :param line_width: 线宽
    :return: img 图片
    '''
    point_x = np.array([[length, 0, 0, 1]]).T
    point_y = np.array([[0, length, 0, 1]]).T
    point_z = np.array([[0, 0, length, 1]]).T
    point = np.array([[0, 0, 0, 1]]).T
    cameraMatrix = np.append(cameraMatrix, np.zeros([3, 1]), 1)
    pro = np.dot(cameraMatrix, np.dot(camerapose, point))
    pro = pro / pro[2, 0]
    pro_x = np.dot(cameraMatrix, np.dot(camerapose, point_x))
    pro_x = pro_x / pro_x[2, 0]
    pro_y = np.dot(cameraMatrix, np.dot(camerapose, point_y))
    pro_y = pro_y / pro_y[2, 0]
    pro_z = np.dot(cameraMatrix, np.dot(camerapose, point_z))
    pro_z = pro_z / pro_z[2, 0]
    img = cv2.line(img, (int(pro[0, 0]), int(pro[1, 0])), (int(pro_x[0, 0]), int(pro_x[1, 0])), (255, 0, 0),
                   thickness=line_width)
    img = cv2.line(img, (int(pro[0, 0]), int(pro[1, 0])), (int(pro_y[0, 0]), int(pro_y[1, 0])), (0, 255, 0),
                   thickness=line_width)
    img = cv2.line(img, (int(pro[0, 0]), int(pro[1, 0])), (int(pro_z[0, 0]), int(pro_z[1, 0])), (0, 0, 255),
                   thickness=line_width)

    return img
