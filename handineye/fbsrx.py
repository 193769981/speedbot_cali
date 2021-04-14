import numpy as np
from utils import tranformUtils as utils
from scipy import optimize as op
def proj_error(Hx,poseList, extrinsicList, real_coor):
    n = len(poseList)
    error = np.array([])
    a = np.size(real_coor, 0)
    real_coor = np.append(real_coor.T, np.zeros([1, a]), 0)
    real_coor = np.append(real_coor, np.ones([1, a]), 0)

    for i in range(n-1):
        Hbhi = np.array(poseList[i])
        Hcgi = np.array(extrinsicList[i])
        a = np.dot(Hcgi,real_coor)
        b = np.dot(Hx,a)
        c = np.dot(Hbhi,b)
        for j in range(i+1,n):
            Hbhj = np.array(poseList[j])
            Hcgj = np.array(extrinsicList[j])
            d = np.dot(np.linalg.inv(Hbhj),c)
            e = np.dot(np.linalg.inv(Hx),d)
            proj = np.dot(np.linalg.inv(Hcgj),e)
            proj[:, :] = proj[:, :] / proj[3, :]
            error = np.append(error, proj[0:3, :] - real_coor[0:3, :])
            T = proj[0:3, :] - real_coor[0:3, :]
    return np.abs(error)
def proj_error2(Hx,poseList, extrinsicList, real_coor):
    n = len(poseList)
    error = np.array([])
    a = np.size(real_coor, 0)
    real_coor = np.append(real_coor.T, np.zeros([1, a]), 0)
    real_coor = np.append(real_coor, np.ones([1, a]), 0)

    for i in range(n-1):
        Hbhi = np.array(poseList[i])
        Hcgi = np.array(extrinsicList[i])
        a = np.dot(Hcgi,real_coor)
        b = np.dot(Hx,a)
        c = np.dot(Hbhi,b)
        for j in range(i+1,n):
            Hbhj = np.array(poseList[j])
            Hcgj = np.array(extrinsicList[j])
            d = np.dot(np.linalg.inv(Hbhj),c)
            e = np.dot(np.linalg.inv(Hx),d)
            proj = np.dot(np.linalg.inv(Hcgj),e)
            proj[:, :] = proj[:, :] / proj[3, :]
            error = np.append(error,np.mean(np.abs(proj[0:3, :] - real_coor[0:3, :])))
            T = proj[0:3, :] - real_coor[0:3, :]

    return np.abs(error)

def loss_function(X,poseList, extrinsicList, real_coor):
    R = utils.eulerAngles2RotationMatrix(X[:3])
    T = X[3:6]
    H = np.append(R, np.array([T]).T, 1)
    H = np.append(H, np.array([[0, 0, 0, 1]]), 0)
    return proj_error(H,poseList,extrinsicList,real_coor)


def refine(init,poseList, extrinsicList, real_coor):
    init_euler = utils.rotationMatrix2EulerAngles(init[:3,:3])
    init = np.append(init_euler,init[:3,3])
    # a =np.size(real_coor,0)
    # real_coor = np.append(real_coor.T, np.zeros([1,a]),0)
    # real_coor = np.append(real_coor,np.ones([1,a]),0)
    solver = op.root(loss_function,init, args=(poseList, extrinsicList, real_coor),method='lm')
    R = utils.eulerAngles2RotationMatrix(solver.x[:3])
    T = solver.x[3:6]
    H = np.append(R, np.array([T]).T, 1)
    H = np.append(H, np.array([[0, 0, 0, 1]]), 0)
    return H