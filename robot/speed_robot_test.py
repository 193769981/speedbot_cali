#-*- coding:utf-8 -*-

from robot import robot_base
from socket import socket
import transforms3d
import numpy as np
import math

class robot(robot_base.robot):

    def __init__(self, host, port, trans_method):
        self.host = host
        self.port = port
        self.trans_method = trans_method
        self.socket = socket()
        self.initpose=np.identity(4)

    def get_init_pose(self):
        '''
        给机器人发送'current_pose'指令，得到初始的robot_pose
        :return:
        '''
        print('host and port')
        print(self.host)
        print(self.port)
        try:
            self.socket.connect((self.host, self.port))
        except:
            flag = False
            print('robot ip and port error!!!')
            return flag, None, None
        order = 'current_pose,'
        self.socket.send(order.encode())
        init_pose_str = self.socket.recv(60240).decode("utf8", "ignore")
        print(init_pose_str)
        init_pose_str_list = init_pose_str.split(',')
        init_pose = []
        for i in range(len(init_pose_str_list)):
            if i<6:
                init_pose.append(float(init_pose_str_list[i]))
        # init_pose = [float(pose) for pose in init_pose_str_list]
        print("init_pose",init_pose)
        # position first, angle behind
        # init_pose_str -- angle
        init_pose_str_re = '@,' + str(init_pose[3]) + ',' + str(init_pose[4]) + ',' + str(init_pose[5]) + ',' + str(init_pose[0]) + ',' + str(init_pose[1]) + ',' + str(init_pose[2])
        if init_pose[0] == None:
            flag = False
            return flag, None, None
        else:
            flag = True
            # 欧拉角
            R = transforms3d.euler.euler2mat(init_pose[0] * math.pi / 180, init_pose[1] * math.pi / 180, init_pose[2] * math.pi / 180, self.trans_method)
            print(R)
            T = np.array([init_pose[3], init_pose[4], init_pose[5]]).reshape([3, 1])
            pose = np.identity(4)
            pose[:3, :3] = R
            for i in range(3):
                pose[i, 3] = T[i][0] / 1000
            # pose[:3, 3] = T
            self.initpose=pose
            return flag, init_pose_str_re, pose

    def init_move(self, init_robot_pose):
        self.socket.send(init_robot_pose.encode())
        recv = self.socket.recv(60240).decode("utf8")
        print("receive:" + recv)
        if recv != '':
            flag = True
        else:
            flag = False
        return flag, recv

    def moveable(self, pose,z_qt):
        '''
        给机器人发送位姿信息，检查是否可达
        :param pose: 要移动位姿的理论值 -- 4*4matrix
        :return:flag(表示是否可达)
        '''
        print('moveable z' , z_qt)
        r = transforms3d.euler.mat2euler(pose[:3, :3], self.trans_method)
        t = pose[:3, 3].flatten()
        initz = self.initpose[2][3]
        if (t[2] < initz-0.1 or t[2] > initz + float(z_qt)):  # ABB 600  1200
            # print(".....come in constrain z.....")
            return False
        print("#,{},{},{},{},{},{}".format(round(t[0] * 1000, 3), round(t[1] * 1000, 3), round(t[2] * 1000, 3), round(r[0] * 180 / math.pi, 3), round(r[1] * 180 / math.pi, 3), round(r[2] * 180 / math.pi, 3)))
        self.socket.send("#,{},{},{},{},{},{}".format(round(t[0] * 1000, 3), round(t[1] * 1000, 3), round(t[2] * 1000, 3), round(r[0] * 180 / math.pi, 3), round(r[1] * 180 / math.pi, 3), round(r[2] * 180 / math.pi, 3)).encode())

        flag = self.socket.recv(1024).decode()
        if flag[0] == 't':
            return True
        else:
            print("robot false")
            return False

    def move_mat(self, pose):
        '''
        给机器人发送位姿信息，得到最终实际的robot_pose
        :param pose: 要移动位姿的理论值 -- 4*4matrix
        :return: flag+移动位姿的实际值(-- 4*4matrix)
        '''
        print('moveable not z')
        r = transforms3d.euler.mat2euler(pose[:3, :3], self.trans_method)
        t = pose[:3, 3].flatten()
        print("#,{},{},{},{},{},{}".format(round(t[0] * 1000, 3), round(t[1] * 1000, 3), round(t[2] * 1000, 3),
                                           round(r[0] * 180 / math.pi, 3), round(r[1] * 180 / math.pi, 3),
                                           round(r[2] * 180 / math.pi, 3)))
        self.socket.send("@,{},{},{},{},{},{}".format(round(t[0] * 1000, 3), round(t[1] * 1000, 3), round(t[2] * 1000, 3), round(r[0] * 180 / math.pi, 3), round(r[1] * 180 / math.pi, 3), round(r[2] * 180 / math.pi, 3)).encode())
        recv = self.socket.recv(60240).decode("utf8")
        print('recv')
        print(recv)
        pose_str_list = recv.split(',')
        pose_list = []
        for i in range(len(pose_str_list)):
            if i<6:
                pose_list.append(float(pose_str_list[i]))
        # pose_list = [float(pose) for pose in pose_str_list]
        flag = pose_str_list[0]

        print("valid send and receive")
        print("send")
        print(round(t[0] * 1000, 3),round(t[1] * 1000, 3),round(t[2] * 1000, 3))
        print("receive")
        print(pose_list[3],pose_list[4],pose_list[5])

        if not flag:
            return False, None, None
        else:
            R = transforms3d.euler.euler2mat(pose_list[0] * math.pi / 180, pose_list[1] * math.pi / 180, pose_list[2] * math.pi / 180, self.trans_method)
            T = np.array([pose_list[3], pose_list[4], pose_list[5]]).reshape([3, 1])
            pose = np.identity(4)
            pose[:3, :3] = R
            for i in range(3):
                pose[i, 3] = T[i][0] / 1000
            return True, pose, recv

    def move_eul(self, pose):
        '''
        给机器人发送位姿信息，得到最终实际的robot_pose
        :param pose: 要移动位姿的理论值
        :return: flag+移动位姿的实际值(euler欧拉角+位移形式)
        '''
        # self.socket.connect((self.host, self.port))
        r = transforms3d.euler.mat2euler(pose[:3, :3], self.trans_method)
        t = pose[:3, 3].flatten()
        self.socket.send("@,{},{},{},{},{},{}".format(round(t[0] * 1000, 3), round(t[1] * 1000, 3), round(t[2] * 1000, 3), round(r[0] * 180 / math.pi, 3), round(r[1] * 180 / math.pi, 3), round(r[2] * 180 / math.pi, 3)).encode())
        recv = self.socket.recv(60240).decode("utf8")
        print(recv)
        if recv == '':
            return False, None
        else:
            return True, recv


    def release(self):
        order = "end"
        self.socket.send(order.encode())
        self.socket.close()