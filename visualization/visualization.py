import numpy as np
import os
#import open3d as o3d


def visualization(pointcld_data, m_error, file_path, colors_data=None):
    '''
    存储成点云文件，进行可视化展示
    :param pointcld_data:点云位置数据
    :param m_error:该可视化图像的平均误差，以此命名
    :param file_path: 保存的文件的路径
    :param colors_data: 点云颜色数据
    '''
    # point_cloud = o3d.PointCloud()
    # point_cloud.points = o3d.Vector3dVector(pointcld_data)
    # if colors_data is not None:
    #     point_cloud.colors = o3d.Vector3dVector(colors_data)

    # # del_list = os.listdir(file_path)
    # # for f in del_list:
    # #     del_path = file_path
    # #     if os.path.isfile():
    # o3d.write_point_cloud(file_path +'{}.pcd'.format(m_error), point_cloud)
    # obj = o3d.read_point_cloud(file_path +'{}.pcd'.format(m_error))
    #o3d.draw_geometries([obj])
    # o3d.draw_geometries([point_cloud])
    print(1)