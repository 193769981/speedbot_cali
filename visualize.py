# -*- coding:utf-8 -*-
import numpy as np
from visualization import visualization
#import open3d as o3d
import os
import datetime


def visualize(real_shot_coor_list, real_marker_coor_list, proj2cam_list, extrinsic_list, rz_error):
    # 判断是否存在pcd文件夹
    # if not os.path.exists('pcd/'):
    #     os.makedirs('pcd/')
    # current_time = "-".join(str(datetime.datetime.now()).split())
    # file_path = 'pcd/{}/'.format(current_time)
    # os.makedirs(file_path)
    # # 进行可视化对比
    # for i in range(len(real_shot_coor_list)):
    #     # 蓝色的点，相机拍到的真实的三维点
    #     comp = real_shot_coor_list[i]
    #     colors = [[0, 0, 1] for point in range(len(real_shot_coor_list[i]))]

    #     # 红色的点，通过手眼矩阵还原得到的三维点
    #     comp = np.append(comp, proj2cam_list[i].T[:, 0:3], 0)
    #     colors.extend([[1, 0, 0] for point in range(len(proj2cam_list[i].T))])

    #     # 绿色的点，通过相机外参还原得到的三维点
    #     # points_m2c = np.dot(extrinsic_list[i], np.append(real_marker_coor_list[i].T, np.ones([1, real_marker_coor_list[i].shape[0]]), 0))
    #     # comp = np.append(comp, points_m2c.T[:, 0:3], 0)
    #     # colors.extend([[0, 1, 0] for point in range(len(points_m2c.T))])
    #     visualization.visualization(comp, rz_error[i], file_path, colors)

    # # 进行误差分析
    # axis_pcd = o3d.create_mesh_coordinate_frame(size=0.005, origin=[0, 0, 0])

    # error_points = []
    # for image in range(len(proj2cam_list)):
    #     for point in range(proj2cam_list[image].shape[1]):
    #         error_points.append(real_shot_coor_list[image][point] - proj2cam_list[image].T[:, 0:3][point])

    # # print(error_points)
    # point_cloud = o3d.PointCloud()
    # point_cloud.points = o3d.Vector3dVector(error_points)
    print(1)
    #o3d.draw_geometries([axis_pcd] + [point_cloud])
    file_path = '1'
    return file_path