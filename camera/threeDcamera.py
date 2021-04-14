#-*- coding:utf-8 -*-
# @time:
# @author:liminhao
# @email: minhao.lee@qq.com
from camera import camera_base
from socket import socket
import cv2
from utils.load_data import load_data
import time

class threeD(camera_base.camera):
    def __init__(self, client, intrinsic, dist):
        self.temp_color_file_name = None
        self.temp_depth_file_name = None
        self.img_RootPath = None
        self.type = '3d'
        self.intrinsic = intrinsic
        self.dist = dist
        self.client = client
        self.imgsize = None

    def get_rgb_depth_image(self):
        time.sleep(2)
        self.client.send("externsic|photograph|1".encode('utf-8'))
        #print("externsic|photograph|1")
        img_RootPath = self.client.recv(1024).decode()
        print('get depth and rgb img',img_RootPath)
        if img_RootPath:
            self.img_RootPath = img_RootPath
            data = {'DepthMap': [], 'PointCloudZ': []}
            data_path, imgName_list, robot_pose = load_data(img_RootPath, data, imgName_list=[], robot_pose=[])
            self.temp_color_file_name = imgName_list[-1]
            self.temp_depth_file_name = data_path['PointCloudZ'][-1]
            rgb_image = cv2.imread(self.temp_color_file_name)
            depth_image = cv2.imread(self.temp_depth_file_name, -1)
            self.imgsize = (rgb_image.shape[1], rgb_image.shape[0])
            return True, rgb_image, depth_image, img_RootPath
        else:
            return False, None, None, None

    # useless temporarily
    def release(self):
        self.client.send("end".encode())
        self.client.close()
        return
    def sendProcessBar(self,str):
        self.client.send(str.encode('utf-8'))