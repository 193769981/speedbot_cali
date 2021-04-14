#-*- coding:utf-8 -*-
# @time: 
# @author:张新新
# @email: 1262981714@qq.com

from abc import ABCMeta, abstractmethod

class robot(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def move(self, pose):
        '''
        指挥机械臂进行运动
        :return pose:机器人实际到达位姿
        '''
        pass

    @abstractmethod
    def moveable(self, pose):
        '''
        返回pose是否在可达空间中
        :param pose:
        :return:
        '''

    @abstractmethod
    def release(self):
        '''
        释放资源
        :return:
        '''
