from utils import load_data
from AprilTag.aprilTagUtils import *
import cv2



def calibrate(board, img_path):
    imglist = []
    robot_pose = []
    data = {}

    _, imgName_list, _ = load_data.load_data(img_path, data, imglist, robot_pose)
    imglist = []
    for i in imgName_list:
        imglist.append(cv2.imread(i))
    img_size = tuple([list(imglist[0].shape)[1], list(imglist[0].shape)[0]])

    # 开始检测标定板角点,得到内参和畸变系数
    # 利用了cv2.calibrateCamera标定内参和畸变参数
    ret, _, _, camera_matrix, discoff = camera_cali(board, img_size, imglist, 0)
    print("内参标定平均重投影误差为(pixel)：{}".format(ret))

    return camera_matrix, discoff