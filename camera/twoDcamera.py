#-*- coding:utf-8 -*-
# @time: 
# @author:liminhao
# @email: minhao.lee@qq.com
from camera import camera_base
from socket import socket
import cv2
from utils.load_data import load_data

class twoD(camera_base.camera):
    def __init__(self, client, intrinsic, dist):
        # self.host = host
        # self.port = port
        self.client = client
        # self.socket = socket()
        self.temp_color_file_name = None
        # self.temp_depth_file_name = None
        self.img_RootPath = None
        self.type = '2d'
        self.intrinsic = intrinsic
        self.dist = dist
        self.imgsize = None

    def get_rgb_image(self):
        print('get rgb image')
        # self.socket.connect((self.host, self.port))
        # 给界面发消息,触发相机拍照
        self.client.send("externsic|photograph|1".encode('utf-8'))
        img_RootPath = self.client.recv(1024).decode()
        print('img_RootPath',img_RootPath)
        if img_RootPath:
            self.img_RootPath = img_RootPath
            data = {'DepthMap': [], 'PointCloudZ': []}
            data_path, imgName_list, robot_pose = load_data(img_RootPath, data, imgName_list=[], robot_pose=[])
            # print('img list',imgName_list)
            self.temp_color_file_name = imgName_list[-1]
            rgb_image = cv2.imread(self.temp_color_file_name)
            self.imgsize = (rgb_image.shape[1], rgb_image.shape[0])
            # self.socket.close()
            return True, rgb_image,img_RootPath
        else:
            # self.socket.close()
            return False, None,None

    # useless temporarily
    def release(self):
        self.client.send("end".encode())
        self.client.close()
        return

    def sendProcessBar(self,str):
        self.client.send(str.encode('utf-8'))