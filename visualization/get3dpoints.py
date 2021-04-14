import numpy as np
#import open3d as o3d

def depth2xyz(depth_map, camera_matrix, imgsize, flatten=False, depth_scale=1000, disrete=False):
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    if disrete == False:
        z = depth_map / depth_scale
        y, x = np.mgrid[0:list(imgsize)[1], 0:list(imgsize)[0]]
        x = (x - cx) * z / fx
        y = (y - cy) * z / fy
        #print(list(imgsize)[1])
        #print('x',x)
        coor = np.dstack((x, y, z)) if flatten == False else np.dstack((x, y, z)).reshape(-1, 3)
    else:
        z = depth_map[:, 2] / depth_scale
        depth_map[:, 0] = (depth_map[:, 0] - cx) * z / fx
        depth_map[:, 1] = (depth_map[:, 1] - cy) * z / fy
        coor = np.dstack((depth_map[:, 0], depth_map[:, 1], z)) if flatten == False else np.dstack((depth_map[:, 0], depth_map[:, 1], z)).reshape(-1, 3)
    return coor


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