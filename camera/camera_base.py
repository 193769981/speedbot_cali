#-*- coding:utf-8 -*-
# @time: 
# @author:张新新
# @email: 1262981714@qq.com
from abc import ABCMeta,abstractmethod

class camera(object):
    __metaclass__ = ABCMeta
    @abstractmethod
    def get_rgb_image(self):
        '''
        get a image
        :return: image
        '''
        pass
    def get_rgb_depth_image(self):
        '''
        get a image
        :return: image
        '''
        return None, None

    @abstractmethod
    def release(self):
        '''
        释放资源
        :return:
        '''
