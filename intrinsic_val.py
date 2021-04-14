from visualization import get3dpoints
from utils import load_data
from utils import depthUtils
from AprilTag.aprilTagUtils import *
import io
import yaml


def validate(board, intrinsic, discoff, rgb_img_path, dep_img_path):
    # read image data
    dep_img = cv2.imread(dep_img_path, -1)
    rgb_img = cv2.imread(rgb_img_path)
    
    dep_img_size = tuple([list(dep_img.shape)[1], list(dep_img.shape)[0]])
    img_size = tuple([list(rgb_img.shape)[1], list(rgb_img.shape)[0]])
    assert dep_img_size == img_size, "RGB图片与深度图片不一致!"

    # detect tags
    tags = detectTags_img(board, rgb_img, verbose=0)
    corners = [tag.corners for tag in tags]
    assert len(tags) == len(corners), "检测有误,请重新拍照!"

    shot_coor = []
    for seq, corner in enumerate(corners):
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

        shot_coor.append(
            [corner[0][0], corner[0][1], dep_img[int(round(corner[0][1])), int(round(corner[0][0]))]])
        shot_coor.append(
            [corner[1][0], corner[1][1], dep_img[int(round(corner[1][1])), int(round(corner[1][0]))]])
        shot_coor.append(
            [corner[2][0], corner[2][1], dep_img[int(round(corner[2][1])), int(round(corner[2][0]))]])
        shot_coor.append(
            [corner[3][0], corner[3][1], dep_img[int(round(corner[3][1])), int(round(corner[3][0]))]])

    assert shot_coor != [], "未检测到角点，请重新加载图片！"

    shot_coor = np.asarray(shot_coor)
    shot_coor_tmp = np.array([item for item in shot_coor])
    xyz = get3dpoints.depth2xyz(shot_coor_tmp, intrinsic, img_size, flatten=True, disrete=True)

    marker_coor = []
    for tag in tags:
        _, real_marker_corner = board.getPointsbyTagId(tag.tag_id)
        marker_coor.extend(real_marker_corner)
    marker_coor = np.asarray(marker_coor)
    n = np.size(marker_coor, 0)
    marker_coor = np.append(marker_coor, np.zeros([n, 1]), 1)

    extrinsic_list = depthUtils.get_camerapose_by_depth([marker_coor], [xyz])
    externsic = extrinsic_list[0]
    print("-----------------------------delimit-------------------------------")
    # reproject to camera
    validation_points = np.dot(externsic, np.append(marker_coor.T, np.ones([1, marker_coor.T.shape[1]]), axis=0))[0:3, :]
    # reproject to image
    validation_points = np.dot(intrinsic, validation_points)
    validation_points[:, :] = validation_points[:, :] / validation_points[2, :]

    error = np.linalg.norm(validation_points.T[:, 0:2] - shot_coor[:, 0:2], axis=0)
    error_mean = np.mean(error)

    return error_mean