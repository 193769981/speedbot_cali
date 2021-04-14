import numpy as np

def get_camerapose_by_depth(src_list, dst_list):
    '''
    将marker板坐标系上的点，与深度图还原得到的点进行一一对应。利用奇异值分解得到转换矩阵，即外参。
    :param src_list:要进行转换的源坐标列表（marker板坐标系）
    :param dst_list: 被转换到的目的坐标列表（相机坐标系下拍摄到的点）
    :return: 转换矩阵列表（外参列表）
    '''
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


def get_camerapose_by_depth_one(src, dst):
    '''
    将marker板坐标系上的点，与深度图还原得到的点进行一一对应。利用奇异值分解得到转换矩阵，即外参。
    :param src_list:要进行转换的源坐标列表（marker板坐标系）
    :param dst_list: 被转换到的目的坐标列表（相机坐标系下拍摄到的点）
    :return: 转换矩阵列表（外参列表）
    '''
    P = src
    Q = dst
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

    return extrinsic