import os
import cv2
import re

def load_data(ImgPath, data_path, imgName_list, robot_pose):
    for item in os.listdir(ImgPath):
        if os.path.isdir(os.path.join(ImgPath, item)):
            for dir_name in data_path.keys():
                if dir_name == item:
                    img_file = os.listdir(os.path.join(ImgPath, item))
                    img_file.sort(key=lambda num: int(re.findall(r'\d+', num)[0]))
                    for img in img_file:
                        # img_data = cv2.imread(os.path.join(ImgPath, dir_name, img), -1)
                        data_path[dir_name].append(os.path.join(ImgPath, dir_name, img))
            if item == 'RGB':
                img_file = os.listdir(os.path.join(ImgPath, item))
                img_file.sort(key=lambda num: int(re.findall(r'\d+', num)[0]))
                for img in img_file:
                    # img_data = cv2.imread(os.path.join(ImgPath, item, img))
                    imgName_list.append(os.path.join(ImgPath, item, img))
        elif item.endswith('robotpose.txt'):
            pose = open(os.path.join(ImgPath, item))
            lines = pose.readlines()
            for line in lines:
                tmp = line.split()
                if tmp == []:
                    continue
                robot_pose.append(tmp)
    return data_path, imgName_list, robot_pose