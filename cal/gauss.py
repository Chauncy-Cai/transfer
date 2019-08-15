#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/8/10 12:54
#@Author: csc
#@File  : gauss.py

import scipy.stats as stats
import numpy as np

def getGaussPossiblityVec():
    '''
    Guass distribution: mean 0 and D= 1.5
    :return: vec from -4~+4
    '''
    possibility = [stats.norm(0,1.5).pdf(i-4) for i in range(9)]
    possibility = np.array(possibility)
    possibility = possibility/np.sum(possibility)
    return possibility

def getGuassPossiblityTensor():
    vec = getGaussPossiblityVec()
    matrix = np.zeros((9,9,9))
    for i in range(9):
        for j in range(9):
            for k in range(9):
                matrix[i][j][k] = vec[i]*vec[j]*vec[k]
    return matrix

def blockReplace(matrix256,position,matrix9):
    '''
    need a 256*256*256 block,and replace
    a 9by9by9 nearby input position
    :param position:
    :return:
    '''
    def indexDetermine(x):#support
        if 0 > x - 4:  # abnormal
            xmin = 0
            bxmin = 4 - x
        else:
            xmin = x - 4
            bxmin = 0
        if 256 <= x + 4:
            xmax = 256
            bxmax = 9 + 255-(x+4)
        else:
            xmax = x + 4 + 1
            bxmax  = 9
        #print(xmin,xmax,bxmin,bxmax)
        return xmin,xmax,bxmin,bxmax
    #print("position:",position)
    x,y,z = position
    xmin,xmax,bxmin,bxmax = indexDetermine(x)
    ymin, ymax, bymin, bymax = indexDetermine(y)
    zmin, zmax, bzmin, bzmax = indexDetermine(z)
    matrix256[xmin:xmax,ymin:ymax,zmin:zmax] += matrix9[bxmin:bxmax,bymin:bymax,bzmin:bzmax]
    #print(matrix256[xmin:xmax,ymin:ymax,zmin:zmax].shape)
    #print(matrix9[bxmin:bxmax,bymin:bymax,bzmin:bzmax].shape)
    return matrix256

tensor1 = np.zeros((256,256,256))
block = np.ones((9,9,9))
tensor1 = blockReplace(tensor1,[0,0,0],block)
tensor1 = blockReplace(tensor1,[255,255,255],block)
tensor1 = blockReplace(tensor1,[100,100,100],block)
tensor1 = blockReplace(tensor1,[252,4,100],block)
