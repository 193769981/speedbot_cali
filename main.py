# -*- coding: utf-8 -*-
# coding: unicode_escape
import visualize
import cal_handtoeye_matrix
import cal_handineye_matrix
import intrinsic_cal
import intrinsic_val
import test_points_on_robot
import follow_up_calibrate
from AprilTag.board import apriltagBoard
from utils import tranformUtils
import datetime
import os
from socket import *
import threading
import socket
from threading import Thread
import socketserver
import numpy as np
import cv2
import robot_socket_test
import time
import faulthandler
from auto_handeye_real_world import *

# root_path = os.getcwd() + '/'
root_path = '/auto_calibration_python'+ '/'
board_global = None
pointStr =""

# 标定配置
def board(if_input, b_path, type, april_family, marker_X, marker_Y, markerSeparation, tag_size, tagID, board_name):
    '''
    如果用户选择输入标定板参数，则执行此函数；
    若不输入参数，则直接将用户选中的文件路径转发给下个模块
    :param if_input_board: 0-直接转发路径， 1-输入标定板信息
    :param b_path: 已经保存的标定板路径(if_input_board==0)
    :param type: 标定板类型
    :param april_family:
    :param marker_X: markerx方向tag个数
    :param marker_Y: markery方向tag个数
    :param markerSeparation: tag之间的间距
    :param tag_size: tag的边长
    :param tagID: 标定板的ID

    :param board_name: 保存的名字
    :return: 标定板配置文件路径
    '''
    try:
        if if_input == 0:
            return b_path
        else:
            if type == 'AprilTag':
                dir_path = "config/{}_board_para/".format(type)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                fs = cv2.FileStorage(dir_path + board_name + ".yml", cv2.FILE_STORAGE_WRITE)
                fs.write('april_family', april_family)
                fs.write('marker_X', int(marker_X))
                fs.write('marker_Y', int(marker_Y))
                fs.write('markerSeparation', float(markerSeparation))
                fs.write('tag_size', float(tag_size))
                fs.write('tagID', tagID)
                fs.release()
                return root_path + dir_path + board_name + ".yml"

            elif type == 'chessboard':
                dir_path = "config/{}_board_para/".format(type)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                pass

            elif type == 'circle':
                dir_path = "config/{}_board_para/".format(type)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                pass

            elif type == 'acuro':
                dir_path = "config/{}_board_para/".format(type)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                fs = cv2.FileStorage(dir_path + board_name + ".yml", cv2.FILE_STORAGE_WRITE)
                fs.write('marker_X', int(marker_X))
                fs.write('marker_Y', int(marker_Y))
                fs.write('markerSeparation', float(markerSeparation))
                fs.write('tag_size', float(tag_size))
                fs.write('tagID', tagID)
                fs.release()

                return root_path + dir_path + board_name + ".yml"
    except Exception as e:
        print(e)
        return "board configuration error"


# 内参标定
def intrinsic(way_to_intrinsic, i_path, intrinsic_elements, discoff_elements, board_path, img_path,
              inparameter_save_name):
    '''
    用户可以直接选择以前的配置文件，加载数据；
    如果用户选择输入内参和畸变参数，则保存成配置文件；
    若不输入，则利用算法进行标定，再保存成配置文件。
    :param way_to_intrinsic: int 确定内参和畸变参数的方式：0-选择配置文件，1-手动输入，2-现场标定
    :param i_path: 内参及畸变参数路径(way_to_intrinsic==0)
    :param intrinsic_elements: 字符串 fx|fy|cx|cy(way_to_intrinsic==1)
    :param discoff_elements: 字符串 k1|k2|p1|p2|k3(way_to_intrinsic==1)
    :param board_path:标定板配置文件路径(绝对路径)(way_to_intrinsic==2)
    :param img_path:标定图片路径(way_to_intrinsic==2)
    :param inparameter_save_name: 保存配置文件自定义名称(way_to_intrinsic==1,2)
    :return:内参存放路径，畸变参数存放路径
    '''
    global board_global
    if board_global == None:
        assert board_path != None, "未配置标定板！"
        board_global = apriltagBoard(board_path)
    else:
        pass

    # 直接返回已有路径，无需保存
    if way_to_intrinsic == 0:
        return i_path
    # 需要保存新路径
    else:
        ins_dire_name = 'cali_result/intrinsic/'
        if not os.path.exists(ins_dire_name):
            os.makedirs(ins_dire_name)
        # 手动输入
        if way_to_intrinsic == 1:
            # 以下为相机自带内参及畸变参数
            # intrinsic_elements = [2236.7, 2235.91, 1030.07, 802.554999]
            # discoff = [[-0.124878, 0.153655, -8.00999e-05, -1.12299e-05, -0.00947199]]
            intrinsic_elements = intrinsic_elements.strip().split('|')
            intrinsic_elements = [float(element) for element in intrinsic_elements]
            [fx, fy, cx, cy] = intrinsic_elements
            intrinsic = [[fx, 0, cx],
                         [0, fy, cy],
                         [0, 0, 1]]

            discoff_elements = discoff_elements.strip().split('|')
            discoff = [[float(element) for element in discoff_elements]]
        # 现场标定
        elif way_to_intrinsic == 2:
            # 目前来看用该方法标定出来内参和相机自带内参还存在一定差距
            intrinsic, discoff = intrinsic_cal.calibrate(board_global, img_path)

        # -------配置文件保存-------
        # 如果不自定义命名就按照时间命名
        if inparameter_save_name == '':
            current_time = "-".join(str(datetime.datetime.now()).split())
            inparameter_save_name = current_time

        i_path = root_path + ins_dire_name + "intrinsic_" + inparameter_save_name + ".yml"

        fs = cv2.FileStorage(i_path, cv2.FILE_STORAGE_WRITE)
        fs.write("intrinsic", np.array(intrinsic))
        fs.write("discoff", np.array(discoff))
        fs.release()

        return i_path


# 内参验证
def intrinsic_validate(board_path, i_path, rgb_img_path, dep_img_path):
    global board_global
    if board_global == None:
        assert board_path != None, "未配置标定板！"
        board_global = apriltagBoard(board_path)
    else:
        pass
    fs = cv2.FileStorage(i_path, cv2.FILE_STORAGE_READ)
    intrinsic = fs.getNode('intrinsic').mat()
    discoff = fs.getNode('discoff').mat()
    fs.release()
    error_mean = intrinsic_val.validate(board_global, intrinsic, discoff, rgb_img_path, dep_img_path)
    error_mean = int(error_mean)
    result = "重投影误差为{}像素".format(error_mean)

    return result


def externsic(on_flag, board_path, trans_method, i_path, img_path, handeye_save_name,client,camtype):
    '''
    :param on_flag: 是否为眼在手上：EyeToHand-眼在手外，EyeInHand-眼在手上
    :param board_path: 标定板配置文件路径
    :param trans_method: 欧拉角旋转方式
    :param i_path: 内参及畸变参数路径(相对路径)
    :param img_path: 图片路径(相对路径)
    :param handeye_save_name: 手眼矩阵存放路径(相对路径)
    :return: 手眼矩阵x存放路径,各图片的误差rz_error,存储误差对比图.pcd文件的目录
    '''


    global board_global
    if board_global == None:
        assert board_path != None, "未配置标定板！"
        board_global = apriltagBoard(board_path)
    else:
        pass

    x_path = "cali_result/handeye_matrix/" + "hadeye_" + handeye_save_name + ".yml"
    if not os.path.exists("cali_result/handeye_matrix/"):
        os.makedirs("cali_result/handeye_matrix/")

    fs = cv2.FileStorage(i_path, cv2.FILE_STORAGE_READ)
    intrinsic = fs.getNode('intrinsic').mat()
    discoff = fs.getNode('discoff').mat()
    fs.release()
    print('intrinsic')
    print(intrinsic,discoff)

    if camtype == '3d':
        from camera.threeDcamera import threeD as camera_obj
    if camtype == '2d':
        from camera.twoDcamera import twoD as camera_obj
    camera = camera_obj(client, intrinsic=intrinsic, dist=discoff)

    print('come in cali-hand')
    if on_flag == 'EyeInHand':
        # 眼在手上标定
        shot, marker, proj2cam, extrinsic, x, rz_error = cal_handineye_matrix.calibrate(board_global, trans_method,
                                                                                        img_path,
                                                                                        intrinsic, discoff,camera)

    elif on_flag == 'EyeToHand':
        # 眼在手外标定
        shot, marker, proj2cam, extrinsic, x, rz_error = cal_handtoeye_matrix.calibrate(board_global, trans_method,
                                                                                        img_path,
                                                                                        intrinsic, discoff,camera)

    else:
        pass

    # 转化成四元数
    R = x[:3, :3]
    q = tranformUtils.rot2quat(R)
    T = x[:3, 3].T
    q_t = np.append(T, q, axis=0)
    print("手眼矩阵转化为四元数为（偏移量+四元数）{}".format(q_t))

    fs = cv2.FileStorage(x_path, cv2.FILE_STORAGE_WRITE)
    fs.write("externsic", x)
    fs.write("quater_trans", q_t)
    fs.release()

    file_path = visualize.visualize(shot, marker, proj2cam, extrinsic, rz_error)
    file_path = root_path + file_path
    x_path = root_path + x_path

    return x_path, rz_error, file_path

    # except Exception as e:
    #     return "手眼矩阵标定异常！请检查！"


# 走点验证
def robot_verfig(board_path, i_path, x_path, rgb_img_path, dep_img_path):
    '''
    :param board_path: 标定板配置文件路径
    :param i_path: 内参及畸变参数存放的路径
    :param x_path: 手眼矩阵存放的路径
    :param rgb_img_path: RGB图片的路径
    :param dep_img_path: 深度图片的路径
    :return:已标记的图片路径，机器人坐标系下各角点的坐标
    '''
    global board_global
    if board_global == None:
        assert board_path != None, "未配置标定板！"
        board_global = apriltagBoard(board_path)
    else:
        pass

    # 将标定板置于相机视野下的某处，另外拍一张图片，测试机器人的走点是否准确
    coordinate_results, img_results = test_points_on_robot.test_points(board_global, rgb_img_path, dep_img_path, x_path,i_path)
    coordinate_results = root_path + coordinate_results
    img_results = root_path + img_results

    return coordinate_results, img_results


# 读取点位
def getPose(robot_file):
    temp = np.loadtxt(robot_file)
    poseList = []
    for i in range(temp.shape[0]):
        # r = transforms3d.quaternions.quat2mat(np.array([temp[i,6], temp[i,3],temp[i, 4],temp[i,5]]))
        poseList.append(temp[i, :])
    return poseList


# 自动走点
def AutoGetPose(r_ip, r_port, client, board_flag, board_path, robot_flag, camera_flag, configFileName, cali_type, i_path,x_qt,y_qt,z_qt,angle_qt,minz_qt,maxz_qt):
    '''
    :param r_ip: 机器人ip
    :param r_port: 机器人port
    :param board_flag: apriltag|aurco|circle|chessboard
    :param robot_flag: rxyz|sxyz|rzyz|...
    :param camera_flag: 2d|3d
    :param configFileName:
    :return:
    '''
    import auto_handeye_real_world as auto

    if board_flag == 'apriltag':
        from board.apriltagboard import AprilTagBoard as board_obj
    elif board_flag == "aucro":
        pass
    elif board_flag == "circle":
        pass
    elif board_flag == "chessboard":
        pass

    from robot.speed_robot_test import robot as robot_obj

    if camera_flag == '3d':
        from camera.threeDcamera import threeD as camera_obj
    if camera_flag == '2d':
        from camera.twoDcamera import twoD as camera_obj
    print('board obj', board_path)
    board = board_obj(board_path)
    robot = robot_obj(host=r_ip, port=r_port, trans_method=robot_flag)

    # 连接界面,通过界面连接相机
    fs = cv2.FileStorage(i_path, cv2.FILE_STORAGE_READ)
    intrinsic = fs.getNode('intrinsic').mat()
    discoff = fs.getNode('discoff').mat()
    fs.release()
    if camera_flag == '3d':
        camera = camera_obj(client, intrinsic=intrinsic, dist=discoff)
    if camera_flag == '2d':
        camera = camera_obj(client, intrinsic=intrinsic, dist=discoff)
    print('cali_type',cali_type)
    if cali_type == 'EyeInHand':
        auto_cali = auto.auto_handeye_calibration(board, robot, camera, configFileName, 0, minz_qt,maxz_qt,angle_qt)
        auto_cali.set_select_method = 3
    elif cali_type == 'EyeToHand':
        auto_cali = auto.auto_handeye_calibration(board, robot, camera, configFileName, 1, minz_qt,maxz_qt,angle_qt)
        auto_cali.set_select_method = 3

    #验证精度
    # flag, rgb, depth, img_RootPath = camera.get_rgb_depth_image()
    # x_path="cali_result/handeye_matrix/" + "handeye_config" + ".yml"
    # img_results = test_points_on_robot.test_points2(board, rgb, depth, x_path,i_path)
    # img_results = root_path + img_results
    # print(img_results)
    # str=input("cin")

    res = auto_cali.run(x_qt,y_qt,z_qt)
    if res == "success":
        print('return success',auto_cali.Hx)
        R = auto_cali.Hx[:3, :3]
        q = tranformUtils.rot2quat(R)
        T = auto_cali.Hx[:3, 3].T
        q_t = np.append(T, q, axis=0)
        print("手眼矩阵转化为四元数为（偏移量+四元数）{}".format(q_t))
        x_path = "/auto_calibration_python/cali_result/handeye_matrix/" + "handeye_" + 'AutoSave' + ".yml"

        if not os.path.exists("/auto_calibration_python/cali_result/handeye_matrix/"):
            os.makedirs("/auto_calibration_python/cali_result/handeye_matrix/")
        fs = cv2.FileStorage(x_path, cv2.FILE_STORAGE_WRITE)
        fs.write("externsic", auto_cali.Hx)
        fs.write("quater_trans", q_t)
        fs.write('error',auto_cali.error)
        fs.release()
        return True,auto_cali.Hx,auto_cali.error
    elif res == "invalid robotpose":
        print('return failed')
        raise Exception("请重新选取机器人初始位置!")
        return False,None,None
def AutoGetPose2(r_ip, r_port, client, board_flag, board_path, robot_flag, camera_flag, configFileName, cali_type, i_path,x_qt,y_qt,z_qt,angle_qt,minz_qt,maxz_qt):
    '''
    :param r_ip: 机器人ip
    :param r_port: 机器人port
    :param board_flag: apriltag|aurco|circle|chessboard
    :param robot_flag: rxyz|sxyz|rzyz|...
    :param camera_flag: 2d|3d
    :param configFileName:
    :return:
    '''
    import auto_handeye_real_world as auto

    if board_flag == 'apriltag':
        from board.apriltagboard import AprilTagBoard as board_obj
    elif board_flag == "aucro":
        pass
    elif board_flag == "circle":
        pass
    elif board_flag == "chessboard":
        pass

    from robot.speed_robot_test import robot as robot_obj

    if camera_flag == '3d':
        from camera.threeDcamera import threeD as camera_obj
    if camera_flag == '2d':
        from camera.twoDcamera import twoD as camera_obj
    print('board obj', board_path)
    board = board_obj(board_path)
    robot = robot_obj(host=r_ip, port=r_port, trans_method=robot_flag)

    # 连接界面,通过界面连接相机
    fs = cv2.FileStorage(i_path, cv2.FILE_STORAGE_READ)
    intrinsic = fs.getNode('intrinsic').mat()
    discoff = fs.getNode('discoff').mat()
    fs.release()
    if camera_flag == '3d':
        camera = camera_obj(client, intrinsic=intrinsic, dist=discoff)
    if camera_flag == '2d':
        camera = camera_obj(client, intrinsic=intrinsic, dist=discoff)
    print('cali_type',cali_type)
    if cali_type == 'EyeInHand':
        auto_cali = auto.auto_handeye_calibration(board, robot, camera, configFileName, 0, minz_qt,maxz_qt,angle_qt)
        auto_cali.set_select_method = 3
    elif cali_type == 'EyeToHand':
        auto_cali = auto.auto_handeye_calibration(board, robot, camera, configFileName, 1, minz_qt,maxz_qt,angle_qt)
        auto_cali.set_select_method = 3

    res = auto_cali.run2(x_qt,y_qt,z_qt)
    if res == "success":
        print('return success',auto_cali.Hx)
        R = auto_cali.Hx[:3, :3]
        q = tranformUtils.rot2quat(R)
        T = auto_cali.Hx[:3, 3].T
        q_t = np.append(T, q, axis=0)
        print("手眼矩阵转化为四元数为（偏移量+四元数）{}".format(q_t))
        x_path = "/auto_calibration_python/cali_result/handeye_matrix/" + "handeye_" + 'AutoSave' + ".yml"

        if not os.path.exists("/auto_calibration_python/cali_result/handeye_matrix/"):
            os.makedirs("/auto_calibration_python/cali_result/handeye_matrix/")
        fs = cv2.FileStorage(x_path, cv2.FILE_STORAGE_WRITE)
        fs.write("externsic", auto_cali.Hx)
        fs.write("quater_trans", q_t)
        fs.write('error',auto_cali.error)
        fs.release()
        return True,auto_cali.Hx,auto_cali.error
    elif res == "invalid robotpose":
        print('return failed')
        raise Exception("请重新选取机器人初始位置!")
        return False,None,None
def SemiAutoGetPose(r_ip, r_port, client, board_flag, board_path, robot_flag, camera_flag, configFileName, cali_type, i_path,x_qt,y_qt,z_qt,angle_qt,minz_qt,maxz_qt,posetxtpath):
    '''
    :param r_ip: 机器人ip
    :param r_port: 机器人port
    :param board_flag: apriltag|aurco|circle|chessboard
    :param robot_flag: rxyz|sxyz|rzyz|...
    :param camera_flag: 2d|3d
    :param configFileName:
    :return:
    '''
    import auto_handeye_real_world as auto

    if board_flag == 'apriltag':
        from board.apriltagboard import AprilTagBoard as board_obj
    elif board_flag == "aucro":
        pass
    elif board_flag == "circle":
        pass
    elif board_flag == "chessboard":
        pass

    from robot.speed_robot_test import robot as robot_obj

    if camera_flag == '3d':
        from camera.threeDcamera import threeD as camera_obj
    if camera_flag == '2d':
        from camera.twoDcamera import twoD as camera_obj
    print('board obj', board_path)
    board = board_obj(board_path)
    robot = robot_obj(host=r_ip, port=r_port, trans_method=robot_flag)

    # 连接界面,通过界面连接相机
    fs = cv2.FileStorage(i_path, cv2.FILE_STORAGE_READ)
    intrinsic = fs.getNode('intrinsic').mat()
    discoff = fs.getNode('discoff').mat()
    fs.release()
    if camera_flag == '3d':
        camera = camera_obj(client, intrinsic=intrinsic, dist=discoff)
    if camera_flag == '2d':
        camera = camera_obj(client, intrinsic=intrinsic, dist=discoff)
    print('cali_type',cali_type)
    if cali_type == 'EyeInHand':
        auto_cali = auto.auto_handeye_calibration(board, robot, camera, configFileName, 0, minz_qt,maxz_qt,angle_qt)
        auto_cali.set_select_method = 3
    elif cali_type == 'EyeToHand':
        auto_cali = auto.auto_handeye_calibration(board, robot, camera, configFileName, 1, minz_qt,maxz_qt,angle_qt)
        auto_cali.set_select_method = 3

    res = auto_cali.semirun(x_qt,y_qt,z_qt,posetxtpath)
    if res == "success":
        print('return success',auto_cali.Hx)
        return True,auto_cali.Hx,auto_cali.error
        #pass
    elif res == "invalid robotpose":
        print('return failed')
        raise Exception("请重新选取机器人初始位置!")
        return False,None,None

def AutoGetPose_2DPlane(r_ip, r_port, client, board_flag, board_path, robot_flag, camera_flag, configFileName, cali_type, i_path):
    '''
    :param r_ip: 机器人ip
    :param r_port: 机器人port
    :param board_flag: apriltag|aurco|circle|chessboard
    :param robot_flag: rxyz|sxyz|rzyz|...
    :param camera_flag: 2d|3d
    :param configFileName:
    :return:
    '''
    print('auto get pose 2')
    import auto_handeye_real_world as auto
    if board_flag == 'apriltag':
        from board.apriltagboard import AprilTagBoard as board_obj
    elif board_flag == "aucro":
        pass
    elif board_flag == "circle":
        pass
    elif board_flag == "chessboard":
        pass

    from robot.speed_robot_test import robot as robot_obj

    if camera_flag == '3d':
        from camera.threeDcamera import threeD as camera_obj
    if camera_flag == '2d':
        from camera.twoDcamera import twoD as camera_obj

    board = board_obj(board_path)
    robot = robot_obj(host=r_ip, port=r_port, trans_method=robot_flag)

    # 连接界面,通过界面连接相机
    fs = cv2.FileStorage(i_path, cv2.FILE_STORAGE_READ)
    intrinsic = fs.getNode('intrinsic').mat()
    discoff = fs.getNode('discoff').mat()
    fs.release()
    camera = camera_obj(client, intrinsic=intrinsic, dist=discoff)

    if cali_type == 'EyeInHand':
        auto_cali = auto.auto_handeye_calibration(board, robot, camera, configFileName, 0)
        auto_cali.set_select_method = 3
    elif cali_type == 'EyeToHand':
        auto_cali = auto.auto_handeye_calibration(board, robot, camera, configFileName, 1)
        auto_cali.set_select_method = 3

    #验证精度
    # flag, rgb, depth, img_RootPath = camera.get_rgb_depth_image()
    x_path="cali_result/handeye_matrix/" + "handeye_config" + ".yml"
    # img_results = test_points_on_robot.test_points2(board, rgb, depth, x_path,i_path)
    # img_results = root_path + img_results
    # print(img_results)
    # str=input("cin")
    fs = cv2.FileStorage(x_path, cv2.FILE_STORAGE_READ)
    Hx = fs.getNode('externsic').mat()
    #Hx=np.eye(4)
    print('get3dPointfrom2dcam')
    # 0-cali test   1-cali   2-line camera test 3-line camera cali
    res = auto_cali.get3dPointfrom2dcam(Hx,2)
from socket import *
socket # <class 'socket.socket'>
import threading

def SendPosForRobot(ip, port, client, PosPath, flag, openandcloe):
    # ip = '127.0.0.1' # 获取机器人IP
    # port = 10004  # 设置端口
    s = socket(AF_INET, SOCK_STREAM)
    s.connect((ip, port))
    print("flag is ",flag)
    if flag==1:
        PosPath = str(PosPath + "/robotpose.txt")
        poseList = getPose(PosPath)  # 初始化机器人的pos
        for pose in poseList:
            s.send("@,{0},{1},{2},{3},{4},{5}".format(pose[0], pose[1], pose[2], pose[3], pose[4],
                                                    pose[5]).encode())  # 将位置发给机器人
            print("send true")
            data_robot = s.recv(60240).decode("utf8", "ignore")
            print(data_robot)
            dataSend = "externsic|photograph|1"
            client.send(dataSend.encode("utf-8"))
            time.sleep(5)
            # if data_robot == 'f1':
            #     dataSend = "externsic|photograph|1"
            #     client.send(dataSend.encode('utf-8')
            # # print(dataSend)
            # elif data_robot == "0":
            #     dataSend = "externsic|photograph|0"
            #     client.send(dataSend.encode('utf-8'))
            # s.close()
        s.send("!".encode())
        return True
    elif flag==2:
        print("flag is 2 ",flag)
        s.send("#,".encode())
        data_robot_point = s.recv(1024).decode("utf-8","ignore")
        RobotPointArr = data_robot_point.split(",")
        print("RobotPointArr",RobotPointArr)
        if len(RobotPointArr):
            pointStr = " ".join(RobotPointArr)
            print("pointStr",pointStr)
            pointStr = "12 12 12 12 12 12"
            client.send(("SaveRobotPoint|True|"+pointStr).encode("utf8"))
            '''
            with open(PosPath+"/robotpose.txt","a") as f:
                f.write(str(pointStr))
                f.write("\n")
                # f.close()
            '''
        else:
            print("list is null")
            s.close()
            return False
    # s = socket.socket()
    # s.connect((ip, port))
    s.send("end".encode())
    s.close()
    return True


def handle(client):
    while True:
        data = client.recv(602400).decode('utf-8')
        #print(data)
        if data != "":
            dataArry = data.split("|")
            print(data)
            for i in dataArry:
                print(i)
            if dataArry[0] == "CalBoardClass":  # 测试标定板参数设置
                print("dataArry[0] == CalBoardClass 标定板参数设置")
                if dataArry[1] == "1":  # 手动填
                    # board_path = board(1, '', 'AprilTag', "tag36h11", 7, 5, 0.007776, 0.030385, 'tagId3', 'AprilTag3')
                    board_path = board(1, '', dataArry[2], "tag36h11", dataArry[3], dataArry[4], dataArry[5],
                                       dataArry[6],
                                       dataArry[7], dataArry[8])
                print("board_path", board_path)
                # 回传数据
                reBoard = "CalBoardClass|" + board_path
                client.send(reBoard.encode("utf8"))
                break
            if dataArry[0] == "Internal":  # 内参标定
                print("dataArry[0] == Internal 内参标定")
                if dataArry[1] == "0":  # 手动
                    if dataArry[2] == "0":
                        i_path = intrinsic(0, dataArry[3], '',
                                           '', dataArry[4], '', '')  ##异常处理－－如果传的文件里面没有图片或者图片无法进行内参标定－－返回错误
                        print("Internal", "0", "0")
                    elif dataArry[2] == "1":  # 手动输入
                        datastr1 = dataArry[3] + "|" + dataArry[4] + "|" + dataArry[5] + "|" + dataArry[6]
                        datastr2 = dataArry[7] + "|" + dataArry[8] + "|" + dataArry[9] + "|" + dataArry[10] + "|" + \
                                   dataArry[11]
                        print(datastr1)
                        print(datastr2)
                        i_path = intrinsic(1, '', datastr1,
                                           datastr2, dataArry[12], 'Data/data6', dataArry[13])
                        print("Internal", "0", "1")
                    reipath = "Internal|0|1|" + i_path
                    print("i_path", reipath)
                    client.sendall(reipath.encode('utf-8'))
                elif dataArry[1] == "1":  # 自动标定
                    i_path = intrinsic(2, '', '',
                                       '', dataArry[2], dataArry[3],
                                       dataArry[4])  # 'Data/data6' 标定文件路径　speed_photoneo　生成的文件名称
                    reipath = "Internal|1|" + i_path
                    print("i_path", reipath)
                    client.sendall(reipath.encode('utf-8'))
                break
            if dataArry[0] == "externsic":  # 外参标定
                print("dataArry[0] == externsic 外参标定")
                for i in range(19):
                    dataArry.append('i')
                fs = cv2.FileStorage(dataArry[1], cv2.FILE_STORAGE_READ)
                dataArry[1]=fs.getNode('Calibration_type').string()
                dataArry[3]=fs.getNode('Euler').string()
                dataArry[4]=fs.getNode('RobotIp').string()
                dataArry[5]=fs.getNode('CalClassPath').string()
                dataArry[6]=fs.getNode('InternalPath').string()
                dataArry[7]=fs.getNode('SaveFileName').string()
                dataArry[9]=fs.getNode('CameraIs3D').string()
                dataArry[10]=fs.getNode('X').string()
                dataArry[11]=fs.getNode('Y').string()
                dataArry[12]=fs.getNode('Z').string()
                dataArry[13]=fs.getNode('Angle').string()
                dataArry[14]=fs.getNode('Min').string()
                dataArry[15]=fs.getNode('Max').string()
                dataArry[16]=fs.getNode('CalError').string()
                dataArry[17]=fs.getNode('AutoModel').string()
                dataArry[18]=fs.getNode('SemiAutoCalDataPath').string()
                dataArry[19]=fs.getNode('RobotPort').string()
                if dataArry[1] == "眼在手外":
                    dataArry[1] = "EyeToHand"
                elif dataArry[1] == "眼在手上":
                    dataArry[1] = "EyeInHand"
                if dataArry[9]=="0":
                    dataArry[9]="3d"
                elif dataArry[9]=="1":
                    dataArry[9]="2d"
                fs.release()
                print('dataArray',dataArry)
                # AutoGetPose(r_ip, r_port, board_flag, board_path, robot_flag, camera_flag, configFileName, cali_type, i_path)
                if(dataArry[17]=='Auto'):
                    print("进入自动标定")
                    print(dataArry[4])
                    print(int(dataArry[19]))
                    flag, x, error = AutoGetPose(dataArry[4], int(dataArry[19]), client, 'apriltag', dataArry[5], dataArry[3], dataArry[9], '/auto_calibration_python/auto_set_to.yml', dataArry[1], dataArry[6],dataArry[10],dataArry[11],dataArry[12],dataArry[13],dataArry[14],dataArry[15])
                    print(flag,'quit auto get pose')
                    R = x[:3, :3]
                    q = tranformUtils.rot2quat(R)
                    T = x[:3, 3].T
                    q_t = np.append(T, q, axis=0)
                    print("手眼矩阵转化为四元数为（偏移量+四元数）{}".format(q_t))
                    x_path = "/auto_calibration_python/cali_result/handeye_matrix/" + "handeye_" + dataArry[7] + ".yml"

                    if not os.path.exists("/auto_calibration_python/cali_result/handeye_matrix/"):
                        os.makedirs("/auto_calibration_python/cali_result/handeye_matrix/")
                    fs = cv2.FileStorage(x_path, cv2.FILE_STORAGE_WRITE)
                    fs.write("externsic", x)
                    fs.write("quater_trans", q_t)
                    fs.release()
                    print("autogetpose end")
                elif(dataArry[17]=='Auto2'):
                    print("进入自动标定模式２")
                    print(dataArry[4])
                    print(int(dataArry[19]))
                    flag, x, error = AutoGetPose2(dataArry[4], int(dataArry[19]), client, 'apriltag', dataArry[5],
                                                 dataArry[3], dataArry[9], '/auto_calibration_python/auto_set_to.yml',
                                                 dataArry[1], dataArry[6], dataArry[10], dataArry[11], dataArry[12],
                                                 dataArry[13], dataArry[14], dataArry[15])
                    print(flag, 'quit auto get pose')
                    R = x[:3, :3]
                    q = tranformUtils.rot2quat(R)
                    T = x[:3, 3].T
                    q_t = np.append(T, q, axis=0)
                    print("手眼矩阵转化为四元数为（偏移量+四元数）{}".format(q_t))
                    x_path = "/auto_calibration_python/cali_result/handeye_matrix/" + "handeye_" + dataArry[7] + ".yml"

                    if not os.path.exists("/auto_calibration_python/cali_result/handeye_matrix/"):
                        os.makedirs("/auto_calibration_python/cali_result/handeye_matrix/")
                    fs = cv2.FileStorage(x_path, cv2.FILE_STORAGE_WRITE)
                    fs.write("externsic", x)
                    fs.write("quater_trans", q_t)
                    fs.release()
                    print("autogetpose end")
                elif(dataArry[17]=='SemiAuto'):
                    print("进入semi自动标定")
                    flag, x, error = SemiAutoGetPose(dataArry[4], int(dataArry[19]), client, 'apriltag', dataArry[5],
                                                 dataArry[3], dataArry[9], '/auto_calibration_python/auto_set_to.yml',
                                                 dataArry[1], dataArry[6], dataArry[10], dataArry[11], dataArry[12],
                                                 dataArry[13], dataArry[14], dataArry[15],dataArry[18])
                    R = x[:3, :3]
                    q = tranformUtils.rot2quat(R)
                    T = x[:3, 3].T
                    q_t = np.append(T, q, axis=0)
                    print("手眼矩阵转化为四元数为（偏移量+四元数）{}".format(q_t))
                    x_path = "/auto_calibration_python/cali_result/handeye_matrix/" + "handeye_" + dataArry[7] + ".yml"

                    if not os.path.exists("/auto_calibration_python/cali_result/handeye_matrix/"):
                        os.makedirs("/auto_calibration_python/cali_result/handeye_matrix/")
                    fs = cv2.FileStorage(x_path, cv2.FILE_STORAGE_WRITE)
                    fs.write("externsic", x)
                    fs.write("quater_trans", q_t)
                    fs.release()
                    print("semiauto getpose end")
                else:
                    print("进入手动标定")
                    on_flag=dataArry[1]
                    board_path= dataArry[5] #'config/AprilTag_board_para/AprilTag3.yml'
                    trans_method=dataArry[3]
                    i_path=dataArry[6]
                    img_path=dataArry[18]+'/Data'
                    handeye_save_name='NoneAutoResult'

                    x_path, error, file_path = externsic(on_flag,board_path,trans_method,i_path,img_path,handeye_save_name,client,dataArry[9])

                file_path = "cali_result/pcd"
                file_path = root_path + file_path
                if not os.path.exists("/auto_calibration_python/cali_result/pcd/"):
                    os.makedirs("/auto_calibration_python/cali_result/pcd/")
                print("x_path is ",x_path)
                print("file_path is ",file_path)
                meanError = error
                reExternsic = "externsic|" + x_path + "|" + file_path + "|" + str(meanError)
                client.send(reExternsic.encode('utf-8'))
                break
            if dataArry[0] == "robot_verfig":  # 机器人走点验证
                print("dataArry[1] is ", dataArry[1])
                print("dataArry[2] is ", dataArry[2])
                print("dataArry[3] is ", dataArry[3])
                print("dataArry[4] is ", dataArry[4])
                print("dataArry[5] is ", dataArry[5])
                coordinate_results, img_results = robot_verfig(dataArry[1], dataArry[2], dataArry[3], dataArry[4],
                                                               dataArry[5])  # 第４个参数应该是图片路径，不是图片存在的文件夹路径
                print("coordinate_results is", coordinate_results)  # 坐标点位置文件路径
                print("img_results is", img_results)  # 机器人走点图片
                retuStr = "robot_verfig|" + coordinate_results + "|" + img_results
                client.send(retuStr.encode('utf-8'))
                break
            if dataArry[0] == "Internal_verfig":  ##精度验证
                retuInteVer = intrinsic_validate(dataArry[1], dataArry[2], dataArry[3], dataArry[4])
                print(retuInteVer)
                retuInteVerStr = "Internal_verfig|" + retuInteVer
                client.send(retuInteVerStr.encode('utf-8'))
                break
            if dataArry[0] == "SaveRobotPoint":  # 获取机器人当前位置坐标
                #SendPosForRobot(ip, port, client, PosPath, flag, openandclose):
                if SendPosForRobot(dataArry[2], 11000, client, dataArry[1],2,"dataArry[3]"):
                    # print("pointStr is ",pointStr)
                    # client.send("SaveRobotPoint|True|"+pointStr.encode('utf-8'))
                    pass
                else:
                    client.send("SaveRobotPoint|False".encode('utf-8'))
                break
            if dataArry[0] == "Follow":  # 随动标定
                result = follow_up(dataArry[1], dataArry[2], dataArry[3], dataArry[4], dataArry[5],dataArry[6])
                client.send("Follow|"+result+"|"+dataArry[6].encode('utf-8'))
        else:
            client.send(data.encode('utf-8'))
            print("收到的信息为空!")
            break
    print("succeed")
    client.close()
    return ""




if __name__ == '__main__':

    '''*****************socket server******************'''
    tcpSerSock = socket(AF_INET, SOCK_STREAM)
    tcpSerSock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    tcpSerSock.bind(('0.0.0.0', 10003))
    tcpSerSock.listen(10)

    # print(u'我在%s线程中 ' % threading.current_thread().name)  # 本身是主线程
    print('waiting for connecting...')
    while True:
        clientSock, addr = tcpSerSock.accept()
        print(addr)
        print(clientSock)
        print('connected from:', addr)
        t = threading.Thread(target=handle, args=(clientSock,))
        t.start()
        print("start")
        t.join()
        print("join")
    '''*****************socket server******************'''