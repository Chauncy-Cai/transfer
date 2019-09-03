#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/7/23 4:56
#@Author: csc
#@File  : support.py

import cv2
import open3d as o3d
import numpy as np
from sklearn.neighbors import NearestNeighbors

def decompose(index,imageshape):
    l = imageshape[1] #480
    #print("should be 640",l)
    x = index%l #~480
    y = index//l #~640
    return [x,y]###???

def depthImage2PointCloud(img,intrinsic = None):
    #print(intrinsic)
    if intrinsic is None:
        intrinsic = [[577.590698, 0, 318.905426, 0.000000],
                     [0, 578.729797, 242.683609, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]
    intrinsic = np.array(intrinsic)
    height,width = img.shape
    #print(img.shape)
    ys,xs=np.meshgrid(range(height),range(width),indexing='ij')
    vertex = np.zeros([height*width,3])
    image = np.zeros([height*width,3])
    vertex[:, 2] = img.flatten()
    vertex[:, 0] = ((xs - intrinsic[0, 2]) / intrinsic[0, 0]).flatten() * vertex[:, 2]
    vertex[:, 1] = ((ys - intrinsic[1, 2]) / intrinsic[1, 1]).flatten() * vertex[:, 2]

    #Intrinsic = np.array([[1169.621094, 0.000000, 646.295044],
    #                      [0.000000, 1167.105103, 489.927032],
    #                      [0.000000, 0.000000, 1.000000]])
    #kx, ky =1296/640,968/480
    #Intrinsic[0] = Intrinsic[0]*kx
    #Intrinsic[1] = Intrinsic[1]*ky
    #print(img.shape)
    image[:, 2] = 1
    image[:, 0] = xs.flatten()
    image[:, 1] = ys.flatten()
    '''
    loss = 0
    for i in range(len(image)):
        temp = cross_op(image[i]).dot(intrinsic[:3,:3]).dot(np.zeros((3,4))).dot(
            np.array([[vertex[i][0],vertex[i][1],vertex[i][2],1]]).T)
        loss += temp.T.dot(temp)
    print("loss---",loss)
    '''
    return vertex,image


def DepthPicTo3dPoint(path,intrinsic = None):
    #print(path)
    #img = cv2.imread(path,2)/1000
    #2dimension
    if intrinsic is None:
        img = cv2.imread(path, 2) / 1000
        points,_ = depthImage2PointCloud(img)
    else:
        #print('here is 2')
        img = cv2.imread(path,2)
        points, _ = depthImage2PointCloud(img,intrinsic)
    return points,img.shape

def readColor(path,depthshape):
    img = cv2.imread(path)
    reshapesize = (depthshape[1],depthshape[0])
    #print("imgshape",img.shape)
    img = cv2.resize(img,reshapesize)
    return img

def readCameraMatrix(path):
    f = open(path)
    l = f.readline()
    T = []
    while l:
        temp = [float(i) for i in l.split(" ")]
        T.append(temp)
        l = f.readline()
    T = np.array(T)
    return T

def get3dPointCloud(index):
    '''
    this is for test scannet
    '''
    depthpath = "./depth/" + str(index) + ".png"
    colorpath = "./color/" + str(index) + ".jpg"
    posepath = "./pose/" + str(index) + ".txt"
    points,depthshape = DepthPicTo3dPoint(depthpath)
    color = readColor(colorpath,depthshape)
    color = np.array(color).reshape((-1,3))/256
    color = color.T
    color[[0,2],:] = color[[2,0],:]
    color = color.T
    T = readCameraMatrix(posepath)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.array(points))
    pc.colors = o3d.utility.Vector3dVector(np.array(color))
    pc.transform(T)#------------
    return pc

def PointcloudWithColorShow():
    pc1 = get3dPointCloud(0)
    o3d.visualization.draw_geometries([pc1])

def cross_op(x):
    '''
    :param x: 3d vector
    '''
    u,v,w = x[0],x[1],x[2]
    cross_x_half = np.array([
        [0,-w,v],
        [0,0,-u],
        [0,0,0]])
    return cross_x_half-(cross_x_half.T)

def vec2pose(translation_rotation_vec):
    t = translation_rotation_vec[:3]
    r = translation_rotation_vec[3:]
    theta = np.linalg.norm(r, 2)
    if theta < 1e-5:
        R = np.eye(3)
    else:
        k = r / theta
        """ Roduiguez"""
        R = np.cos(theta)*np.eye(3)+np.sin(theta)*cross_op(k)+(1-np.cos(theta))*np.outer(k, k)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def getColor(index):
    '''
    this is for test scannet
    '''
    depthpath = "./depth/" + str(index) + ".png"
    colorpath = "./color/" + str(index) + ".jpg"
    _ ,depthshape = DepthPicTo3dPoint(depthpath)
    #print("imageshape",depthshape)
    color = readColor(colorpath,depthshape)
    return color

def point_matching(pointlist1, pointlist2, color1=None, color2=None, hascolor=0):
    pointlist1 = np.array(pointlist1)
    pointlist2 = np.array(pointlist2)
    if hascolor == 0:
        nbrs = NearestNeighbors(n_neighbors=1, n_jobs=10).fit(pointlist2)
        _, indices = nbrs.kneighbors(pointlist1)
    else:
        point_and_color1 = np.append(pointlist1, color1, axis=1)
        point_and_color2 = np.append(pointlist2, color2, axis=1)
        nbrs = NearestNeighbors(n_neighbors=1, n_jobs=10).fit(point_and_color2)
        _, indices = nbrs.kneighbors(point_and_color1)
    p1 = np.array([pointlist1[i] for i in range(len(pointlist1))])
    p2 = np.array([pointlist2[indices[i][0]] for i in range(len(pointlist1))])
    indices = np.array(indices).T[0]
    return p1, p2, indices

def pack_Rt(R,t):
    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = t
    return T
#PointcloudWithColorShow()

