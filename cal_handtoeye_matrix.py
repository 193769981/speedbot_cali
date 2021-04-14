from handtoeye import motion
from visualization import get3dpoints
from utils import tranformUtils
from utils import depthUtils
from utils import load_data
from method import li
from handtoeye import rz
from AprilTag.aprilTagUtils import *
#import open3d as o3d
import gc
import  time
from AprilTag.board import apriltagBoard


def calibrate(board, trans_method, img_path, intrinsic_input, discoff_input,camera):
    imgName_list = []
    robot_pose = []
    # configuration
    # data = {}
    data = {'DepthMap': [], 'PointCloudZ': []}
    discoff = discoff_input
    camera_matrix = intrinsic_input
    data_path, imgName_list, robot_pose = load_data.load_data(img_path, data, imgName_list, robot_pose)
    if(len(data_path['PointCloudZ']) != len(imgName_list)):
        str="msg|RGB图像和深度图像没有对齐，请检查！"
        camera.sendProcessBar(str)

    if(len(robot_pose) != len(imgName_list)):
        str="msg|机器人位姿数据与图像没有对齐,请检查!"
        time.sleep(1)
        camera.sendProcessBar(str)
    assert len(data_path['PointCloudZ']) == len(imgName_list), "RGB图像和深度图像没有对齐，请检查！"
    assert len(robot_pose) == len(imgName_list), "机器人位姿数据与图像没有对齐,请检查!"
    print('img list size',len(imgName_list))
    # 获取图片size
    img = cv2.imread(imgName_list[0], -1)
    img_size = tuple([list(img.shape)[1], list(img.shape)[0]])

    # 得到机器人位姿列表
    robot_pose_list = tranformUtils.get_robot_pose(robot_pose, trans_method)

    # 存储数据的两个列表
    imglist = []
    pclzlist = []
    # 将tag对象(包含位姿)，以及tag角点的像素坐标记录下来。
    tags_list = []
    # 要删除图片的序号
    reject_ids = []
    # 相机拍摄的三维点云坐标
    real_shot_coor_list = []
    # 标定板真实世界坐标系下的坐标
    real_marker_coor_list = []

    for i in range(len(robot_pose_list)):
        shot_coor = []
        delete_tag_list = []
        img_data = cv2.imread(imgName_list[i])
        pclz = cv2.imread(data_path['PointCloudZ'][i], -1)
        # 图像去畸变
        img_data = cv2.undistort(img_data, camera_matrix, discoff)
        pclz = cv2.undistort(pclz, camera_matrix, discoff)

        imglist.append(img_data)
        pclzlist.append(pclz)

        tags = detectTags_img(board, img_data, camera_matrix, verbose=0)
        tags_list.append(tags)
        corners = [tag.corners for tag in tags]

        for corner in corners:
            delete_flag = 0
            # 若marker板没拍摄完全，计算出来的角点会超过图片边界
            for n in range(4):
                if round(corner[n][1]) >= list(img_size)[1]:
                    print("width超过图片边界({}),已删除该tag".format(corner[n][1]))
                    delete_flag = 1

                if round(corner[n][0]) >= list(img_size)[0]:
                    print("height超过图片边界({}),已删除该tag".format(corner[n][0]))
                    delete_flag = 1

            # 如果有超界的点，则越过,不记录该tag
            if delete_flag:
                continue
            delete_tag_list.append(delete_flag)
            shot_coor.append([corner[0][0], corner[0][1], pclz[int(round(corner[0][1])), int(round(corner[0][0]))]])
            shot_coor.append([corner[1][0], corner[1][1], pclz[int(round(corner[1][1])), int(round(corner[1][0]))]])
            shot_coor.append([corner[2][0], corner[2][1], pclz[int(round(corner[2][1])), int(round(corner[2][0]))]])
            shot_coor.append([corner[3][0], corner[3][1], pclz[int(round(corner[3][1])), int(round(corner[3][0]))]])
        if shot_coor == []:
            reject_ids.append(i)
            continue
        shot_coor = np.asarray(shot_coor)
        xyz = get3dpoints.depth2xyz(shot_coor, camera_matrix, img_size, flatten=True, disrete=True)
        real_shot_coor_list.append(xyz)

        # 删除无记录点对应的tag,更新tagslist
        tmp = []
        for seq, flag in enumerate(delete_tag_list):
            if flag == 0:
                tmp.append(tags[seq])

        tags = tmp

        marker_coor = []
        for tag in tags:
            _, real_marker_corner = board.getPointsbyTagId(tag.tag_id)
            marker_coor.extend(real_marker_corner)
        marker_coor = np.asarray(marker_coor)
        n = np.size(marker_coor, 0)
        marker_coor = np.append(marker_coor, np.zeros([n, 1]), 1)
        real_marker_coor_list.append(marker_coor)
    # 计算得到所有数据的外参
    extrinsic_list = depthUtils.get_camerapose_by_depth(real_marker_coor_list, real_shot_coor_list)

    # 删除没有检测到tag的图片
    reject_ids.sort()
    for index in range(len(reject_ids) - 1, -1, -1):
        del robot_pose_list[reject_ids[index]]
        del tags_list[reject_ids[index]]
        del extrinsic_list[reject_ids[index]]
        del real_shot_coor_list[reject_ids[index]]
    
    # 释放内存
    del imglist
    del pclzlist
    gc.collect()

    print('come in compute.')
    str="msg|come in compute,wait..."
    print(str)
    camera.sendProcessBar(str)

    while (True):
        A, B = motion.motion_axyb(robot_pose_list, extrinsic_list)
        x, y = li.calibration(A, B)

        # real_shot_coor_list中每一项的点的个数可能不为4 * row_num * col_num，需要根据已检测出的角点进行调整
        x, y = rz.refine(x, y, robot_pose_list, extrinsic_list, board, real_shot_coor_list, tags_list)
        rz_error, proj2cam_list = rz.RMSE2cam(x, y, robot_pose_list, board, real_shot_coor_list, tags_list)
        error_max = np.max(rz_error)
        error_mean = np.mean(rz_error)
        if error_max <= 0.001 or len(robot_pose_list) <= 8:
            print("--------------------\n手眼矩阵为：")
            x_acc = x
            for m in range(len(x)):
                for n in range(len(x[m])):
                    x_acc[m][n] = format(x[m][n], '.4f')
            print(x_acc)
            print("--------------------\n各图片中的标定误差为(m)：")
            print(rz_error)
            print("--------------------\n平均误差为(m)：")
            print(error_mean)
            print("--------------------")
            break
        else:
            # 返回numpy的索引，行号和列号
            x_i, y_i = np.where(rz_error.reshape([1, -1]) == error_max)
            # 根据索引号删除图片
            del robot_pose_list[y_i[0]]
            del tags_list[y_i[0]]
            del extrinsic_list[y_i[0]]
            del real_shot_coor_list[y_i[0]]


    return real_shot_coor_list, real_marker_coor_list, proj2cam_list, extrinsic_list, x_acc, rz_error

    # real_shot_coor_list = []
    # real_marker_coor_list = []
    # proj2cam_list = []
    # extrinsic_list = []
    # x_acc = np.random.rand(4,4)
    # rz_error = 1
    # return real_shot_coor_list, real_marker_coor_list, proj2cam_list, extrinsic_list, x_acc, rz_error