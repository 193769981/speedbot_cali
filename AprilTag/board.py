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
import platform
if platform.system().lower() == 'linux':
    import dt_apriltags as apriltag
else:
    import pupil_apriltags as apriltag

class apriltagBoard:
    april_family = "tag36h11"
    marker_X = 7
    marker_Y = 5
    markerSeparation = 0.007776
    tag_size = 0.030385
    tagID = 'tagId1'
    tag_id_order = np.array([])
    boardcenter = np.array([])
    boardcorner = np.array([])
    conners_order = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]])
    # conners_order = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])

    
    def __init__(self, configFile):
        self.getParameter(configFile)
        try:
            print("config/tagId/" + self.tagID + ".csv")
            self.tag_id_order = np.loadtxt("/auto_calibration_python/config/tagId/" + self.tagID + ".csv", delimiter=",")
            # self.tag_id_order = np.loadtxt("config/tagId/" + self.tagID + ".csv", delimiter=",")
        except IOError:
            raise IOError("tagID缺失！请检查！")
        self.get_board_points()
        # 调用apriltag库函数
        # print('april_family',self.april_family)
        self.at_detector = apriltag.Detector(families=self.april_family)
        
        
        
    def getParameter(self, configfile):

        fs = cv2.FileStorage(configfile, cv2.FILE_STORAGE_READ)
        self.marker_X = int(fs.getNode("marker_X").real())
        self.marker_Y = int(fs.getNode("marker_Y").real())
        self.markerSeparation = fs.getNode("markerSeparation").real()
        self.tag_size = fs.getNode("tag_size").real()
        self.tagID = str(fs.getNode("tagID").string())
        print(configfile)
        print(fs.getNode("tag_size").real())
        print(self.tagID)
        fs.release()
        

    def get_board_points(self):
        self.boardcenter = np.empty([self.marker_X * self.marker_Y, 2])
        self.boardcorner = np.empty([4 * self.marker_X * self.marker_Y, 2])
        m, n = self.marker_X, self.marker_Y
        l = self.tag_size
        seq = self.markerSeparation
        for i in range(n):
            for j in range(m):
                center_x = j * (l + seq)
                center_y = i * (l + seq)
                self.boardcenter[i * m + j, 0] = center_x
                self.boardcenter[i * m + j, 1] = center_y
                for k in range(4):
                    self.boardcorner[4 * (i * m + j) + k, 0] = center_x + l / 2.0 * self.conners_order[k, 0]
                    self.boardcorner[4 * (i * m + j) + k, 1] = center_y + l / 2.0 * self.conners_order[k, 1]

    def getPointsbyTagId(self, tagId):
        x, y = np.where(self.tag_id_order == tagId)
        assert len(x) != 0 and len(y) != 0, "tagId选取错误，请检查！"
        center = self.boardcenter[x[0] * self.marker_X + y[0], :]
        corner = self.boardcorner[4 * (x[0] * self.marker_X + y[0]):4 * (x[0] * self.marker_X + y[0]) + 4, :]
        return center, corner