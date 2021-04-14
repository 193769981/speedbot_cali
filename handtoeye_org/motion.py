# -*- coding: utf-8 -*-
import numpy as np

def motion_axxb(hb2gs,hc2os):
    '''
    根据Hb2gs和Hc2os获取A和B,用于构建AX=XB
    :param hb2gs: list<array<4*4>> 机器臂姿态
    :param hc2os: list<array<4*4>> 相机外参
    :return: A,B:list<array<4*4>> AX=XB的参数
    '''
    a = []
    b = []
    n = len(hb2gs)
    for i in range(n-1):
        for j in range(i,n):
            a.append(np.dot(hb2gs[j],np.linalg.inv(hb2gs[i])))
            b.append(np.dot(hc2os[j],np.linalg.inv(hc2os[i])))
    return a,b


def motion_axyb(hb2gs,hc2os):
    '''
    根据Hb2gs和Hc2os获取A和B,用于构建AX=YB
    :param hb2gs: list<array<4*4>> 机器臂姿态
    :param hc2os: list<array<4*4>> 相机外参
    :return: A,B:list<array<4*4>> AX=YB的参数
    '''
    a = []
    b = []
    for i in range(len(hb2gs)):
        a.append(np.linalg.inv(hb2gs[i]))
        b.append(np.linalg.inv(hc2os[i]))
    return a, b