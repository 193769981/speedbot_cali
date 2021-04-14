import open3d as o3d
from handtoeye import motion
from scipy import optimize as opt
from handtoeye import rx
import math
import re
from method import li
from method import dual
from handtoeye import rz
from AprilTag.board import apriltagBoard
from AprilTag.aprilTagUtils import *

ImgPath = 'data5'
imglist = []
robot_pose = []
data = {'DepthMap': [], 'PointCloudX': [], 'PointCloudY': [], 'PointCloudZ': []}

for item in os.listdir(ImgPath):
    if os.path.isdir(os.path.join(ImgPath, item)):
        for dir_name in data.keys():
            if dir_name == item:
                img_file = os.listdir(os.path.join(ImgPath, item))
                img_file.sort(key=lambda num: int(re.match(r'\d+', num).group()))
                for img in img_file:
                    img_data = cv2.imread(os.path.join(ImgPath, dir_name, img), -1)
                    data[dir_name].append(img_data)
        if item == 'RGB':
            img_file = os.listdir(os.path.join(ImgPath, item))
            img_file.sort(key=lambda num : int(re.match(r'\d+', num).group()))
            for img in img_file:
                img_data = cv2.imread(os.path.join(ImgPath, item, img))
                imglist.append(img_data)
    elif item.endswith('.txt'):
        pose = open(os.path.join(ImgPath, item))
        lines = pose.readlines()
        for line in lines:
            tmp = line.split()
            robot_pose.append(tmp)


def depth2xyz(depth_map, camera_matrix, flatten=False, depth_scale=1000, disrete=False):
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    if disrete == False:
        z = depth_map / depth_scale
        y, x = np.mgrid[0:1544, 0:2064]
        x = (x - cx) * z / fx
        y = (y - cy) * z / fy
        coor = np.dstack((x, y, z)) if flatten == False else np.dstack((x, y, z)).reshape(-1, 3)
    else:
        z = depth_map[:, 2] / depth_scale
        depth_map[:, 0] = (depth_map[:, 0] - cx) * z / fx
        depth_map[:, 1] = (depth_map[:, 1] - cy) * z / fy
        coor = np.dstack((depth_map[:, 0], depth_map[:, 1], z)) if flatten == False else np.dstack((depth_map[:, 0], depth_map[:, 1], z)).reshape(-1, 3)
    return coor


def visualization(pointcld_data, colors_data=None):
    '''
    :param pointcld_data:点云位置数据
    :param colors_data: 点云颜色数据
    :return: 进行可视化展示
    '''
    point_cloud = o3d.PointCloud()
    point_cloud.points = o3d.Vector3dVector(pointcld_data)
    if colors_data is not None:
        point_cloud.colors = o3d.Vector3dVector(colors_data)
    o3d.draw_geometries([point_cloud])


def get_PointCloudXYZ(data):
    '''
    :param data:有PointCloudXYZ的字典数据。
    :return:将X，Y，Z三个维度的数据拼接到一起
    '''
    for i in range(len(data['PointCloudX'])):
        PointCLoud = np.asarray([[0, 0, 0]])
        for j in range(len(data['PointCloudX'][i])):
            px = np.asarray(data['PointCloudX'][i][j]).reshape(-1, 1)
            py = np.asarray(data['PointCloudY'][i][j]).reshape(-1, 1)
            pz = np.asarray(data['PointCloudZ'][i][j]).reshape(-1, 1)
            PointCLoud_line = np.append(px, py, 1)
            PointCLoud_line = np.append(PointCLoud_line, pz, 1)
            PointCLoud = np.append(PointCLoud, PointCLoud_line, 0)
        # 删除点云为(0,0,0)的点
        delete_list = []
        for k in range(len(PointCLoud)):
            if PointCLoud[k][0] == PointCLoud[k][1] == PointCLoud[k][2] == 0:
                delete_list.append(k)
        PointCLoud = np.delete(PointCLoud, delete_list, 0)
        return PointCLoud


def get_camerapose_by_depth(src_list, dst_list):
    assert len(src_list) == len(dst_list)
    extrinsic_list = []
    for i in range(len(src_list)):
        P = src_list[i]
        Q = dst_list[i]
        meanP = np.mean(P, axis=0)
        meanQ = np.mean(Q, axis=0)
        P_ = P - meanP
        Q_ = Q - meanQ

        W = np.dot(Q_.T, P_)
        U, S, VT = np.linalg.svd(W)
        R = np.dot(U, VT)
        if np.linalg.det(R) < 0:
            R[2, :] *= -1
        T = meanQ.T - np.dot(R, meanP.T)
        RT = np.append(R, T.reshape(3, 1), 1)
        extrinsic = np.append(RT, np.array([[0, 0, 0, 1]]), 0)
        extrinsic_list.append(extrinsic)
    return extrinsic_list

# 机器人位姿
robot_pose_list = []
for pose in robot_pose:
    R = transforms3d.euler.euler2mat(float(pose[3]) * math.pi / 180, float(pose[4]) * math.pi / 180,
                                     float(pose[5]) * math.pi / 180, 'sxyz')
    robot_pose = np.zeros([4, 4])
    robot_pose[:3, :3] = R[:, :]
    # 平移向量
    robot_pose[0, 3] = float(pose[0]) / 1000
    robot_pose[1, 3] = float(pose[1]) / 1000
    robot_pose[2, 3] = float(pose[2]) / 1000
    robot_pose[3, 3] = 1.0
    robot_pose_list.append(robot_pose)

# 开始检测标定板角点
board = apriltagBoard("config/apriltag.yml", "config/tagId2.csv")
imgsize = tuple([2064, 1544])

# 得到的tags_list中没有用到相机内参，最终没用这个数据
_, camera_matrix, discoff = camera_cali(board, imgsize, imglist, 0)
camera_matrix = np.asarray([[2253.91, 0, 1055.62],
                            [0, 2253.13, 777.802],
                            [0, 0, 1]])
# 图像去畸变
for i in range(len(imglist)):
    imglist[i] = cv2.undistort(imglist[i], camera_matrix, discoff)
    data['DepthMap'][i] = cv2.undistort(data['DepthMap'][i], camera_matrix, discoff)

# 得到内参后，重新检测tag，得到tags_list，这样能够获得tag的位姿。
tags_list = []
corners_pixel_list = []
for i in range(len(imglist)):
    tags = detectTags_img(board, imglist[i], camera_matrix, verbose=0)
    tags_list.append(tags)
    tags_pcl = np.asarray([])
    corners = [tag.corners for tag in tags]
    corners_pixel_list.append(corners)

# 构建相机位姿列表
real_shot_coor_list = []
real_marker_coor_list = []
extrinsic_list = []
num = 0
while (True):
    # 相机拍摄的三维点云坐标
    real_shot_coor_list = []
    # 标定板真实世界坐标系下的坐标
    real_marker_coor_list = []
    for i in range(len(data['DepthMap'])):
        shot_coor = []
        for corner in corners_pixel_list[i]:
            shot_coor.append([corner[0][0], corner[0][1], data['DepthMap'][i][round(corner[0][1]), round(corner[0][0])]])
            shot_coor.append([corner[1][0], corner[1][1], data['DepthMap'][i][round(corner[1][1]), round(corner[1][0])]])
            shot_coor.append([corner[2][0], corner[2][1], data['DepthMap'][i][round(corner[2][1]), round(corner[2][0])]])
            shot_coor.append([corner[3][0], corner[3][1], data['DepthMap'][i][round(corner[3][1]), round(corner[3][0])]])
        shot_coor = np.asarray(shot_coor)
        xyz = depth2xyz(shot_coor, camera_matrix, flatten=True, disrete=True)
        real_shot_coor_list.append(xyz)

        marker_coor = []
        for tag in tags_list[i]:
            _, real_marker_corner = board.getPointsbyTagId(tag.tag_id)
            marker_coor.extend(real_marker_corner)
        marker_coor = np.asarray(marker_coor)
        n = np.size(marker_coor, 0)
        marker_coor = np.append(marker_coor, np.zeros([n, 1]), 1)
        real_marker_coor_list.append(marker_coor)
    # 每次循环都会计算，无需进行删除操作
    extrinsic_list = get_camerapose_by_depth(real_marker_coor_list, real_shot_coor_list)

    A, B = motion.motion_axyb(robot_pose_list, extrinsic_list)
    x, y = li.calibration(A, B)
    print("-------------------------------")
    print("优化前")
    print(x)
    print("-------------------------------")

    # real_shot_coor_list中每一项的点的个数可能不为4*5*7，需要根据已检测出的角点进行调整
    x, y = rz.refine(x, y, robot_pose_list, extrinsic_list, board, real_shot_coor_list, tags_list)
    # rz_error, proj2marker_list, proj2cam_list = rz.RMSE2(x, y, robot_pose_list, camera_pose_list, board.boardcorner)
    print("-------------------------------")
    print("优化后")
    print(x)
    print("-------------------------------")
    rz_error, proj2cam_list = rz.RMSE2cam(x, y, robot_pose_list, board, real_shot_coor_list, tags_list)
    error_max = np.max(rz_error)
    error_mean = np.mean(rz_error)
    num += 1

    print(rz_error)
    print(error_mean)
    if error_max < 0.0005:
        break
    else:
        pass

    if len(imglist) == 7:
        break

    if num > 0 and error_max > 0.0005:
        # 返回numpy的索引，行号和列号
        x_i, y_i = np.where(rz_error.reshape([1, -1]) == error_max)

        del robot_pose_list[y_i[0]]
        del imglist[y_i[0]]
        del corners_pixel_list[y_i[0]]
        # del proj2marker_list[y_i[0]]
        del data['DepthMap'][y_i[0]]
        del tags_list[y_i[0]]
        num = 0


# 进行可视化对比
for i in range(len(real_shot_coor_list)):
    # 蓝色的点，相机拍到的真实的三维点
    comp = real_shot_coor_list[i]
    colors = [[0, 0, 1] for point in range(len(real_shot_coor_list[i]))]

    # 红色的点，通过手眼矩阵还原得到的三维点
    comp = np.append(comp, proj2cam_list[i].T[:, 0:3], 0)
    colors.extend([[1, 0, 0] for point in range(len(proj2cam_list[i].T))])

    # 绿色的点，通过相机外参还原得到的三维点
    points_m2c = np.dot(extrinsic_list[i], np.append(real_marker_coor_list[i].T, np.ones([1, real_marker_coor_list[i].shape[0]]), 0))
    comp = np.append(comp, points_m2c.T[:, 0:3], 0)
    colors.extend([[0, 1, 0] for point in range(len(points_m2c.T))])
    # print("外参误差")
    # print(real_shot_coor_list[i] - points_m2c.T[:, 0:3])
    visualization(comp, colors)


# 进行误差分析
axis_pcd = o3d.create_mesh_coordinate_frame(size=0.005, origin=[0, 0, 0])

error_points = []
for image in range(len(proj2cam_list)):
    for point in range(proj2cam_list[image].shape[1]):
        error_points.append(real_shot_coor_list[image][point] - proj2cam_list[image].T[:, 0:3][point])

print(error_points)
print(np.mean(error_points, axis=0))
point_cloud = o3d.PointCloud()
point_cloud.points = o3d.Vector3dVector(error_points)
o3d.draw_geometries([axis_pcd] + [point_cloud])