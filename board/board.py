from abc import ABCMeta,abstractmethod
import numpy as np
import transforms3d
from scipy import optimize as op
import cv2
from matplotlib import pyplot as plt
from board import utils
class board(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def getParameter(self,configfile):
        pass


    @abstractmethod
    def GetBoardAllPoints(self):
        """
        获取标定板上所有可计算的角点
        :return: objpoints: 所有角点
        """
        pass

    @abstractmethod
    def getObjImgPointList(self, image, verbose=0):
        '''
        提取tags中的角点，board的坐标
        :param image: 图片
        :return: objPoint board对应角点
                imgPoint 对应角点在图片上的坐标
        '''
        pass
    @abstractmethod
    def getObjImgPointListFromCircle(self, image,row,col, verbose=0):
        '''
        提取tags中的角点，board的坐标
        :param image: 图片
        :return: objPoint board对应角点
                imgPoint 对应角点在图片上的坐标
        '''
        pass
    @abstractmethod
    def visible(self,campose,intrinsic,dist):
        '''

        :param campose:
        :param intrinsic:
        :param dist:
        :return:
        '''
        pass

    def cameraparam_opt(self,intrinsic, dist, extrinsic_list, imgpoints_list, objpoints_list,expect_rme=0.5, least_pic=10):
        """
        优化相机参数，包括相机内参和相机外参
        :param intrinsic: 相机内参
        :param dist: 相机畸变
        :param extrinsic_list: 相机外参
        :param imgpoints_list: 图片角点列表
        :param objpoints_list: 对应角点坐标
        :param expect_rme: 最大重投影误差，用来筛选掉噪点
        :param least_pic:  最少图片数量
        :return:
        """

        def compose_paramter_vector(A, k, W):
            alpha = np.array([A[0, 0], A[1, 1], A[0, 2], A[1, 2]])
            alpha = np.append(alpha, k)
            P = alpha
            for i in range(len(W)):
                R, t = (W[i])[:3, :3], (W[i])[:3, 3]
                # 旋转矩阵转换为一维向量形式
                zrou = transforms3d.quaternions.mat2quat(R)
                w = np.append(zrou, t)
                P = np.append(P, w)
            return P

        def decompose_paramter_vector(P, lengthdiscoff):
            [alpha, gamma, uc, vc] = P[0:4]
            discoff = P[4:4 + lengthdiscoff]
            A = np.array([[alpha, 0, uc],
                          [0, gamma, vc],
                          [0, 0, 1]])
            W = []
            M = (len(P) - 4 - lengthdiscoff) / 7

            for i in range(int(M)):
                m = 4 + lengthdiscoff + 7 * i
                zrou = P[m:m + 4]
                t = (P[m + 4:m + 7]).reshape(3, -1)

                # 将旋转矩阵一维向量形式还原为矩阵形式
                R = transforms3d.quaternions.quat2mat(zrou)

                # 依次拼接每幅图的外参
                w = np.append(R, t, axis=1)
                w = np.append(w, np.array([[0, 0, 0, 1]]), 0)
                W.append(w)

            return A, discoff, W

        def camera_loss_function(P, discoff_len, imgpoints, objpoints_list):
            intrisic, dist, extrinsic_list = decompose_paramter_vector(P, discoff_len)
            return self.reprojection_error(intrisic, dist, extrinsic_list, imgpoints, objpoints_list)

        discoff_len = np.size(dist)
        P_init = compose_paramter_vector(intrinsic, dist, extrinsic_list)

        solver = op.root(camera_loss_function, P_init, args=(discoff_len, imgpoints_list, objpoints_list), method="lm")
        intrinsic_opt, dist_opt, extrinsic_list_opt = decompose_paramter_vector(solver.x,discoff_len)
        reproject_error = self.reprojection_error(intrinsic_opt, dist_opt, extrinsic_list_opt, imgpoints_list, objpoints_list)
        return reproject_error,intrinsic_opt, dist_opt, extrinsic_list_opt


    def reprojection_error(self,intrisic, dist, extrinsic_list, imgpoints_list,objpoints_list):
        """
        重投影误差
        :param intrisic: 相机内参
        :param dist: 相机畸变
        :param extrinsic_list: 相机外参
        :param imgpoints: 图片角点列表
        :param objpoints_list: 对应角点坐标
        :return:
        """
        errors = np.array([])
        for i in range(len(extrinsic_list)):
            rvec = extrinsic_list[i][:3, :3]
            tvec = extrinsic_list[i][:3, 3]
            real_coor_temp = np.append(objpoints_list[i], np.zeros([objpoints_list[i].shape[0], 1]), 1)
            imagePoints, jacobian = cv2.projectPoints(real_coor_temp, rvec, tvec, intrisic, dist)
            imagePoints = imagePoints.reshape([-1, 2])
            error = imagePoints - imgpoints_list[i]
            errors = np.append(errors, error)
        return errors.flatten()



    def RME_each_pic(self,intrisic, dist, extrinsic_list, imgpoints,objpoints_list):
        """
        计算每幅图片的重投影误差，用于减少误差
        :param intrisic:
        :param dist:
        :param extrinsic_list:
        :param imgpoints:
        :param objpoints_list:
        :return:
        """
        errors = np.array([])
        for i in range(len(extrinsic_list)):
            rvec = extrinsic_list[i][:3, :3]
            tvec = extrinsic_list[i][:3, 3]
            real_coor_temp = np.append(objpoints_list[i], np.zeros([objpoints_list[i].shape[0], 1]), 1)
            imagePoints, jacobian = cv2.projectPoints(real_coor_temp, rvec, tvec, intrisic, dist)
            imagePoints = imagePoints.reshape([-1, 2])
            error = imagePoints - imgpoints[i]
            errors = np.append(errors, np.mean(np.abs(error)))
        return errors

    def intrinsic(self,imgpoints_list, objpoints_list, imgsize):
        """
        相机内参标定
        :param imgpoints_list: 图片角点列表
        :param objpoints_list: 对应标定板上的点
        :param imgsize:
        :return:
        """
        real_coors = []
        img_points = []
        n = len(imgpoints_list)
        for i in range(len(imgpoints_list)):
            l = np.size(imgpoints_list[i], 0)
            a = np.ndarray([l, 1, 3], dtype=np.float32)
            b = np.ndarray([l, 1, 2], dtype=np.float32)
            for j in range(l):
                a[j, 0, 0] = objpoints_list[i][j, 0]
                a[j, 0, 1] = objpoints_list[i][j, 1]
                a[j, 0, 2] = 0
                b[j, 0, 0] = imgpoints_list[i][j, 0]
                b[j, 0, 1] = imgpoints_list[i][j, 1]
            real_coors.append(a)
            img_points.append(b)
        ret, mtx, dist, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, rme = cv2.calibrateCameraExtended(
            real_coors, img_points, imgsize, None, None)
        return rme, mtx, dist


    def extrinsic_opt(self,intrinsic, dist, extrinsic, imgpoints_list, objpoints_list):


        def lossfunction(P, intrinsic, discoff, imgpoints, real_coor):
            R = transforms3d.quaternions.quat2mat(P[:4])
            T = P[4:]
            H = np.append(np.append(R,np.transpose([T]),1),np.array([[0,0,0,1]]),0)
            return self.reprojection_error(intrinsic,discoff,[H] ,[imgpoints], [real_coor])
        P_init = transforms3d.quaternions.mat2quat(extrinsic[:3,:3])
        P_init = np.append(P_init,extrinsic[:3,3])

        solver = op.root(lossfunction, P_init, args=(intrinsic, dist, imgpoints_list, objpoints_list), method="lm")
        P = solver.x
        R = transforms3d.quaternions.quat2mat(P[:4])
        T = P[4:]
        extrinsic_opt = np.append(np.append(R,np.transpose([T]),1),np.array([[0,0,0,1]]),0)

        return extrinsic_opt


    def stereo_calibration(self,imgpoints_list1,objpoints_list1,imgpoints_list2,objpoints_list2,intrinsic1, dist1, intrinsic2, dist2,
                           extrinsic_list1,extrinsic_list2, board_corner,max_rem=0.005, least_pic_num=10):
        """
        双目标定
        :param imgpoints_list1: 相机1对应图片角点列表
        :param objpoints_list1: 相机1对应标定板坐标
        :param imgpoints_list2: 相机2对应图片角点列表
        :param objpoints_list2: 相机2对应标定板坐标
        :param intrinsic1: 相机1内参矩阵
        :param dist1: 相机1畸变系数
        :param intrinsic2: 相机2内参矩阵
        :param dist2: 相机2畸变系数
        :param extrinsic_list1: 相机1外参列表
        :param extrinsic_list2:  相机2外参列表
        :return: rme: 平均重投影误差
                 H: 相机1到相机2的重投影矩阵
        """

        imgpoint1_correspont_list = []
        imgpoint2_correspont_list = []
        objpoints_correspont_list = []
        extrinsic1_correspont_list = []
        extrinsic2_correspont_list = []

        for i in range(len(imgpoints_list1)):
            if extrinsic_list1[i] is None or extrinsic_list2[i] is None:
                continue
            extrinsic1_correspont_list.append(extrinsic_list1[i])
            extrinsic2_correspont_list.append(extrinsic_list2[i])
            imgpoints1 = imgpoints_list1[i]
            imgpoints2 = imgpoints_list2[i]
            objpoints1 = objpoints_list1[i]
            objpoints2 = objpoints_list2[i]
            imgpoints1_correspont = np.array([])
            imgpoints2_correspont = np.array([])
            objpoints_correspont = np.array([])
            for j in range(imgpoints1.shape[0]):
                for k in range(imgpoints2.shape[0]):
                    if objpoints1[j,0]==objpoints2[k,0] and objpoints1[j,1] == objpoints2[k,1]:
                        imgpoints1_correspont = np.append(imgpoints1_correspont,imgpoints1[j,:])
                        imgpoints2_correspont = np.append(imgpoints2_correspont,imgpoints2[k,:])
                        objpoints_correspont = np.append(objpoints_correspont,np.array([objpoints1[j,0],objpoints1[j,1],0]))
                        break
            imgpoints1_correspont = np.reshape(imgpoints1_correspont,[-1,1,2]).astype(np.float32)
            imgpoints2_correspont = np.reshape(imgpoints2_correspont,[-1,1,2]).astype(np.float32)
            objpoints_correspont = np.reshape(objpoints_correspont,[-1,1,3]).astype(np.float32)
            imgpoint1_correspont_list.append(imgpoints1_correspont)
            imgpoint2_correspont_list.append(imgpoints2_correspont)
            objpoints_correspont_list.append(objpoints_correspont)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
            objpoints_correspont_list, imgpoint1_correspont_list,imgpoint2_correspont_list,intrinsic1,dist1,intrinsic2,
            dist2,None, criteria=criteria)
        H = np.append(np.append(R, T, 1), np.array([[0, 0, 0, 1]]), 0)
        realcoor = board_corner.copy()
        if board_corner.shape[1] == 2:
            board_corner = np.append(board_corner,np.zeros([board_corner.shape[0],1]),1)
        if board_corner.shape[1] == 3:
            board_corner = np.append(board_corner,np.ones([board_corner.shape[0],1]),1)
        HH = H.copy()

        def loss_function(X,real_coor,camera_pose_list1, camera_pose_list2):
            t = X[4:]
            q = X[:4]
            R = transforms3d.quaternions.quat2mat(q)
            H = np.append(np.append(R,np.transpose([t]),1),np.array([[0,0,0,1]]),0)
            return self.stereo_rme_mm(H,camera_pose_list1, camera_pose_list2,real_coor)


        while(True):
            init = transforms3d.quaternions.mat2quat(H[:3, :3])
            init = np.append(init, H[:3, 3])
            solver = op.root(loss_function, init, args=(realcoor, extrinsic_list1, extrinsic_list2), method='lm')
            X = solver.x
            t = X[4:]
            q = X[:4]
            R = transforms3d.quaternions.quat2mat(q)
            H = np.append(np.append(R, np.transpose([t]), 1), np.array([[0, 0, 0, 1]]), 0)
            errors = self.stereo_rme_mm_each_group(extrinsic_list1,extrinsic_list2,H,board_corner)
            max_error = np.max(errors)
            if max_error < max_rem:
                break
            if len(extrinsic_list1)<least_pic_num:
                break
            index = np.where(errors == max_error)[0][0]
            del imgpoint1_correspont_list[index]
            del imgpoint2_correspont_list[index]
            del objpoints_correspont_list[index]
            del extrinsic_list1[index]
            del extrinsic_list2[index]
        rme = np.mean(errors)
        return rme,H


    def stereo_rme_mm(self,H,extrinsic_list1,extrinsic_list2,board_corner):
        if board_corner.shape[1] == 2:
            board_corner = np.append(board_corner,np.zeros([board_corner.shape[0],1]),1)
        if board_corner.shape[1] == 3:
            board_corner = np.append(board_corner,np.ones([board_corner.shape[0],1]),1)
        board_corner = board_corner.T
        number = len(extrinsic_list1)
        errors = np.array([])
        for i in range(number):
            proj = np.dot(np.linalg.inv(extrinsic_list2[i]), np.dot(H, np.dot(extrinsic_list1[i], board_corner)))
            proj[:, :] = proj[:, :] / proj[3, :]
            error = proj[:3, :] - board_corner[:3, :]
            errors = np.append(errors,error)
        return errors


    def stereo_rme_mm_each_group(self,extrinsic_list1, extrinsic_list2, H, board_corner):
        if board_corner.shape[1] == 2:
            board_corner = np.append(board_corner, np.zeros([board_corner.shape[0], 1]), 1)
        if board_corner.shape[1] == 3:
            board_corner = np.append(board_corner, np.ones([board_corner.shape[0], 1]), 1)
        board_corner = board_corner.T
        number = len(extrinsic_list1)
        errors = np.array([])
        for i in range(number):
            proj = proj = np.dot(np.linalg.inv(extrinsic_list2[i]), np.dot(H, np.dot(extrinsic_list1[i], board_corner)))
            proj[:, :] = proj[:, :] / proj[3, :]
            error = proj[:3, :] - board_corner[:3, :]
            errors = np.append(errors, np.mean(np.abs(error)))
        return errors

    def intrinsic_depth_opt(self,objPoints_list, imgPoints_list, depth_list, intrinsic, discoff):

        def distance_error(X, lengthdiscoff, objPoints_list, imgPoints_list, depth_list):
            error = np.array([])
            [alpha, gamma, uc, vc] = X[0:4]
            discoff = X[4:4 + lengthdiscoff]
            A = np.array([[alpha, 0, uc],
                          [0, gamma, vc],
                          [0, 0, 1]])
            n = len(objPoints_list)
            for i in range(n):
                Point_cam_cood = cv2.undistortPoints(imgPoints_list[i].reshape(-1,1,2), A, discoff)
                Point_cam_cood = Point_cam_cood.reshape(-1, 2)
                depth_point_acc = np.append(Point_cam_cood, depth_list[i], 1)
                if Point_cam_cood.shape[0] < 5:
                    continue
                for j in range(depth_point_acc.shape[0]):
                    depth_point_acc[j, 0] = Point_cam_cood[j, 0] * depth_point_acc[j, 2]
                    depth_point_acc[j, 1] = Point_cam_cood[j, 1] * depth_point_acc[j, 2]

                plane = utils.get_nice_plane(depth_point_acc)
                for j in range(depth_point_acc.shape[0]):
                    depth_point_acc[j, 2] = plane[0] * depth_point_acc[j, 0] + plane[1] * depth_point_acc[j, 1] + plane[
                        2]
                    # depth_point_acc[j, 0] = Point_cam_cood[j,0]*depth_point_acc[j, 2]
                    # depth_point_acc[j, 1] = Point_cam_cood[j,1]*depth_point_acc[j, 2]
                m = Point_cam_cood.shape[0]
                for a in range(int(m / 2 - 1)):
                    distance1 = np.linalg.norm(objPoints_list[i][a, :] - objPoints_list[i][m - a - 1, :])
                    distance2 = np.linalg.norm(depth_point_acc[a, :] - depth_point_acc[m - a - 1, :])
                    error = np.append(error, np.abs(distance2 - distance1))

            return error


        alpha = np.array([intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]])
        lengthdiscoff = np.size(discoff)
        init = np.append(alpha, discoff)
        solver = op.root(distance_error, init, args=(lengthdiscoff, objPoints_list, imgPoints_list, depth_list),
                          method="lm")
        X = solver.x
        error = distance_error(X, lengthdiscoff, objPoints_list, imgPoints_list, depth_list)
        [alpha, gamma, uc, vc] = X[0:4]
        discoff = np.array([X[4:4 + lengthdiscoff]])
        A = np.array([[alpha, 0, uc],
                      [0, gamma, vc],
                      [0, 0, 1]])
        return A, discoff

    def extrisic_depth(self, objpoint, imgpoint, point_depth, intrinsic, dist):
        if point_depth.shape[0] < 10:
            print("深度缺失比较严重")
            return False, None
        Point_cam_cood = cv2.undistortPoints(imgpoint.reshape(-1, 1, 2), intrinsic, dist)
        Point_cam_cood = Point_cam_cood.reshape(-1, 2)
        point_in_camera = np.append(Point_cam_cood, np.ones([Point_cam_cood.shape[0], 1]), 1)
        point_in_camera[:, :] = np.multiply(point_in_camera,np.repeat(point_depth, 3, axis=1))
        plane = utils.get_nice_plane(point_in_camera)
        error_plane = np.array([])
        for j in range(point_in_camera.shape[0]):
            t = point_in_camera[j, 2]
            point_in_camera[j, 2] = plane[0] * point_in_camera[j, 0] + plane[1] * point_in_camera[j, 1] + plane[2]
            error_plane = np.append(error_plane, t - point_in_camera[j, 2])

        def rigid_transform_3D(A, B):
            assert len(A) == len(B)

            N = A.shape[0]  # total points

            centroid_A = np.mean(A, axis=0)
            centroid_B = np.mean(B, axis=0)

            # centre the points
            AA = A - np.tile(centroid_A, (N, 1))
            BB = B - np.tile(centroid_B, (N, 1))

            # dot is matrix multiplication for array
            H = np.dot(np.transpose(AA), BB)

            U, S, Vt = np.linalg.svd(H)

            # R = Vt.T * U.T
            R = np.dot(Vt.T, U.T)
            # special reflection case
            if np.linalg.det(R) < 0:
                # print "Reflection detected"
                Vt[2, :] *= -1
                R = np.dot(Vt.T, U.T)

            t = -np.dot(R, centroid_A.T) + centroid_B.T
            t = B.T - np.dot(R, A.T)
            t = np.mean(t, axis=1)

            # print t
            BB = np.dot(R, A.T) + np.tile(np.transpose([t]), (1, N))
            error = BB.T - B

            return R, t

        objpoint = np.append(objpoint,np.zeros([objpoint.shape[0],1]),axis=1)
        R, T = rigid_transform_3D(objpoint, point_in_camera)

        camepose = np.append(np.append(R, np.transpose([T]), 1), np.array([[0, 0, 0, 1]]), 0)
        return True,camepose


    def extrinsic(self,imgpoints, objpoints, intrinsic, dist):
        n = objpoints.shape[0]
        realcoor = np.append(objpoints, np.zeros([n, 1]), 1)
        revl, rvec, tvec = cv2.solvePnP(realcoor, imgpoints, intrinsic, dist,cv2.SOLVEPNP_EPNP)
        R = cv2.Rodrigues(rvec)[0]
        return np.append(np.append(R, tvec, 1), np.array([[0, 0, 0, 1]]), 0)


