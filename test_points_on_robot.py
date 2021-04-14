from visualization import get3dpoints
from utils import load_data
from AprilTag.aprilTagUtils import *
import io
import yaml


def test_points(board, rgb_img_path, dep_img_path, x_path, i_path):
    # configuration
    if not os.path.exists('cali_result/validation_points_img/'):
        os.makedirs("cali_result/validation_points_img/")
    if not os.path.exists('cali_result/validation_points_files/'):
        os.makedirs("cali_result/validation_points_files/")
    fs = cv2.FileStorage(x_path, cv2.FILE_STORAGE_READ)
    x = fs.getNode('externsic').mat()
    fs = cv2.FileStorage(i_path, cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode('intrinsic').mat()
    discoff = fs.getNode('discoff').mat()
    fs.release()

    # read image data
    dep_img = cv2.imread(dep_img_path, -1)
    rgb_img = cv2.imread(rgb_img_path)
    img_size = tuple([list(rgb_img.shape)[1], list(rgb_img.shape)[0]])

    # undistort
    rgb_img = cv2.undistort(rgb_img, camera_matrix, discoff)
    dep_img = cv2.undistort(dep_img, camera_matrix, discoff)

    #cv2.imshow("1",rgb_img)
    #cv2.waitKey(0)
    # detect tags
    tags = detectTags_img(board, rgb_img, camera_matrix, verbose=0)
    corners = [tag.corners for tag in tags]

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
    print(len(shot_coor))
    shot_coor = np.asarray(shot_coor)
    shot_coor_tmp = np.array([item for item in shot_coor])
    xyz = get3dpoints.depth2xyz(shot_coor_tmp, camera_matrix, img_size, flatten=True, disrete=True)

    print("-----------------------------delimit-------------------------------")
    validation_points = np.dot(x, np.append(xyz.T, np.ones([1, xyz.T.shape[1]]), axis=0)).T[:, 0:3]
    validation_points_acc = validation_points
    # keep two decimals
    for m in range(len(validation_points)):
        for n in range(len(validation_points[m])):
            validation_points_acc[m][n] = format(validation_points[m][n], '.4f')
    
    print("图片中对应标号的角点在机器人基座坐标系下的坐标（米）")
    coordinate_results_single = []
    for seq, point in enumerate(validation_points_acc):
        coordinate_results_single.append("Point{}.".format(seq + 1) + str(point))
        print("Point{}.".format(seq + 1) + str(point))

    # 将点云存成文件
    fs = cv2.FileStorage("/auto_calibration_python/cali_result/validation_points_files/validation_points.yml",
                            cv2.FILE_STORAGE_WRITE)
    fs.write("points", np.array(validation_points_acc[0]))
    fs.release()
    coordinate_result = "cali_result/validation_points_files/validation_points.yml"

    cv2.putText(rgb_img, str("<-Point"), (int(shot_coor[0, 0]), int(shot_coor[0, 1])), cv2.FONT_HERSHEY_SIMPLEX,
                2, (0, 0, 255), 3)
    cv2.circle(rgb_img, (int(shot_coor[0, 0]), int(shot_coor[0, 1])), 5, (0, 255, 0), -1)
    cv2.imwrite("/auto_calibration_python/cali_result/validation_points_img/validation_points.png", rgb_img)
    img_result = "cali_result/validation_points_img/validation_points.png"

    # cv2.namedWindow("Test Points", cv2.WINDOW_NORMAL)
    # cv2.imshow("Test Points", rgb_img)
    # cv2.waitKey(1000)

    return coordinate_result, img_result


def test_points2(board, rgb_img_path, dep_img_path, x_path, i_path):
    # configuration
    if not os.path.exists('cali_result/validation_points_img/'):
        os.makedirs("cali_result/validation_points_img/")
    if not os.path.exists('cali_result/validation_points_files/'):
        os.makedirs("cali_result/validation_points_files/")
    fs = cv2.FileStorage(x_path, cv2.FILE_STORAGE_READ)
    Hx = fs.getNode('externsic').mat()
    fs = cv2.FileStorage(i_path, cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode('intrinsic').mat()
    discoff = fs.getNode('discoff').mat()
    fs.release()

    # read image data
    dep_img = dep_img_path
    rgb_img = rgb_img_path
    img_size = tuple([list(rgb_img.shape)[1], list(rgb_img.shape)[0]])

    # undistort
    rgb_img = cv2.undistort(rgb_img, camera_matrix, discoff)
    dep_img = cv2.undistort(dep_img, camera_matrix, discoff)

    # detect tags
    tags = detectTags_img(board, rgb_img, camera_matrix, verbose=0)
    corners = [tag.corners for tag in tags]

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
    xyz = get3dpoints.depth2xyz(shot_coor_tmp, camera_matrix, img_size, flatten=True, disrete=True)
    print("xyz shape:", xyz.shape)
    print("-----------------------------delimit-------------------------------")
    validation_points = np.dot(Hx, np.append(xyz.T, np.ones([1, xyz.T.shape[1]]), axis=0)).T[:, 0:3]
    validation_points_acc = validation_points
    # keep two decimals
    for m in range(len(validation_points)):
        for n in range(len(validation_points[m])):
            validation_points_acc[m][n] = format(validation_points[m][n], '.4f')

    print("图片中对应标号的角点在机器人基座坐标系下的坐标（米）")
    coordinate_results_single = []
    for seq, point in enumerate(validation_points_acc):
        coordinate_results_single.append("Point{}.".format(seq + 1) + str(point))
        print("Point{}.".format(seq + 1) + str(point))

    # 将点云存成文件
    # fs = cv2.FileStorage("cali_result/validation_points_files/validation_points.yml",cv2.FILE_STORAGE_WRITE)
    # for i in range(len(validation_points_acc)):
    #     fs.write("points",i,np.array(validation_points_acc[i]))
    # fs.release()

    for i in range(len(validation_points_acc)):
        cv2.circle(rgb_img, (int(shot_coor[i, 0]), int(shot_coor[i, 1])), 1, (0, 255, 0), -1)
        if i%2==0:
            cv2.putText(rgb_img, str(i), (int(shot_coor[i, 0]), int(shot_coor[i, 1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0, 0, 255), 1)
    cv2.imwrite("cali_result/validation_points_img/validation_points.png", rgb_img)
    img_result = "cali_result/validation_points_img/validation_points.png"

    # cv2.namedWindow("Test Points", cv2.WINDOW_NORMAL)
    # cv2.imshow("Test Points", rgb_img)
    # cv2.waitKey(1000)

    return  img_result
