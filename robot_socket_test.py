from socket import *
import numpy as np
def getPose(robot_file):
    temp = np.loadtxt(robot_file)
    poseList = []
    for i in range(temp.shape[0]):
        # r = transforms3d.quaternions.quat2mat(np.array([temp[i,6], temp[i,3],temp[i, 4],temp[i,5]]))
        poseList.append(temp[i,:])
    return poseList

# if __name__ == "__main__":
#     # move_1 = move_i5_class()
#     # move_1.build_shelf()
#     ip = '127.0.0.1' # 获取本地主机名
#     port = 10004  # 设置端口
#     # 绑定端口
#     poseList = getPose("./Data/data6/robotpose.txt")#初始化机器人的ｐｏｓ
#     for pose in poseList:
#         s = socket()
#         s.connect((ip, port))
#         s.send("{0},{1},{2},{3},{4},{5}".format(pose[0],pose[1],pose[2],pose[3],pose[4],pose[5]).encode())#将位置发给机器人
#         print("send true")
#         s.recv(1024).decode()
#         s.close()
#     s = socket()
#     s.connect((ip, port))
#     s.send("end".encode())
#     s.close()


def sendRobotData():
    # move_1 = move_i5_class()
    # move_1.build_shelf()
    ip = '127.0.0.1' # 获取本地主机名
    port = 11000  # 设置端口
    # 绑定端口
    poseList = getPose("./Data/data6/robotpose.txt")#初始化机器人的ｐｏｓ
    for pose in poseList:
        s = socket()
        s.connect((ip, port))
        s.send("{0},{1},{2},{3},{4},{5}".format(pose[0],pose[1],pose[2],pose[3],pose[4],pose[5]).encode())#将位置发给机器人
        print("send true")
        s.recv(1024).decode()
        s.close()
    s = socket()
    s.connect((ip, port))
    s.send("end".encode())
    s.close()
