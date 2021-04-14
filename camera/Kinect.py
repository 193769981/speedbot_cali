#-*- coding:utf-8 -*-
# @time: 
# @author:张新新
# @email: 1262981714@qq.com
from camera import camera_base
from socket import socket
from PIL import Image
import numpy as np
import cv2
class Kinect(camera_base.camera):
    def __init__(self,port):
        self.port = port
        self.socket = socket()
        self.temp_color_file_name = "temp"
        self.temp_depth_file_name = "temp"

    def get_rgb_depth_image(self):
        self.socket.connect(("127.0.0.1", self.port))
        self.socket.send("capture".encode())
        flag = self.socket.recv(1024).decode()
        if flag:
            image = cv2.imread(self.temp_color_file_name)
            depth_image = Image.open(self.temp_depth_file_name)
            depth_image = np.array(depth_image)
            self.socket.close()
            return True, image, depth_image
        else:
            self.socket.close()
            return False, None, None

    def realease(self):
        self.socket.send("end".encode())
        self.socket.close()
        return