#-*- coding:utf-8
#
import cv2
import numpy as np
from board.board import board
import yaml



class ChessBoard(board):

    marker_X = 0
    marker_Y = 0
    square_size= 0

    def __init__(self,configFile):
        super()
        self.GetParameter(configFile)
    
    def GetParameter(self, configfile):
        f = open(configfile, 'r', encoding='utf-8')
        cont = f.read()
        x = yaml.load(cont)
        self.marker_X = x["marker_X"]
        self.marker_Y = x["marker_Y"]
        self.square_size = x["square_size"]
        f.close()

    def GetImageAndObjPoint(self, pic, verbose=0):
        """
        获取图片中的图片检测角点和图片中对应标定板的点
        :param pic: 图片矩阵
        :param verbose: 可视性图片展示
        :return: flag:是否检测成功，
                imgpoints:图片检测的角点
                objpoints:对于标定板上的点
        """

        gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
        succ, pic_coor = cv2.findChessboardCorners(gray, (self.marker_X, self.marker_Y), None)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        if succ:
            # 执行亚像素级角点检测
            corners2 = cv2.cornerSubPix(gray, pic_coor, (11, 11), (-1, -1), criteria)
            if verbose == 1:
                img = cv2.drawChessboardCorners(pic, (self.marker_X, self.marker_Y), pic_coor, succ)
                cv2.namedWindow("chessboard", 0)
                cv2.circle(img, (pic_coor[0, 0, 0], pic_coor[0, 0, 1]), 3, (255, 0, 0), 3)
                cv2.imshow("chessboard", img)
                cv2.waitKey(0)
            imgpoints = np.zeros([self.marker_X * self.marker_Y, 2])
            l, w, g = pic_coor.shape
            for i in range(l):
                x, y = corners2[i].ravel()
                imgpoints[i, 0] = x
                imgpoints[i, 1] = y
            objpoints = np.zeros((self.marker_X * self.marker_Y, 2), np.float32)
            objpoints[:, :2] = np.mgrid[0:self.marker_X, 0:self.marker_Y].T.reshape(-1, 2) * self.square_size
            return succ, imgpoints,objpoints
        else:
            return succ, 0, 0


    def GetBoardAllPoints(self):
        objpoints = np.zeros((self.marker_X * self.marker_Y, 2), np.float32)
        objpoints[:, :2] = np.mgrid[0:self.marker_X, 0:self.marker_Y].T.reshape(-1, 2) * self.square_size
        return objpoints


    def intrinsic(self, imgpoints_list, objpoints_list,imgsize):
        real_coors = []
        img_points = []
        n = len(imgpoints_list)
        l = np.size(imgpoints_list[0], 0)
        for i in range(len(imgpoints_list)):
            a = np.ndarray([l, 1, 3], dtype=np.float32)
            b = np.ndarray([l, 1, 2], dtype=np.float32)
            for j in range(l):
                a[j, 0, 0] = objpoints_list[i][j, 0]
                a[j, 0, 1] = objpoints_list[i][j, 1]
                a[j, 0, 2] = 0
                b[j, 0, 0] = imgpoints_list[i][j, 0]
                b[j, 0, 1] = imgpoints_list[i][j, 1]
            real_coors.append(a)
            img_points.append(b)
        ret, mtx, dist, rvecs, tvec, stdDeviationsIntrinsics, stdDeviationsExtrinsics, rme = cv2.calibrateCameraExtended(
            real_coors, img_points, imgsize, None, None)
        return rme, mtx, dist

    def extrinsic(self,imgpoints, objpoints,intrinsic,dist):
        n = objpoints.shape[0]
        realcoor = np.append(objpoints, np.zeros([n, 1]), 1)
        revl, rvec, tvec = cv2.solvePnP(realcoor, imgpoints, intrinsic, dist)
        R = cv2.Rodrigues(rvec)[0]
        return np.append(np.append(R, tvec, 1), np.array([[0, 0, 0, 1]]), 0)









