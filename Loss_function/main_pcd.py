#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/8/27 4:17
# @Author: csc
# @File  : main_img.py

import random

from optimizer import *
from support import *


def correpondence(pcd1, pcd2, pcdhelper, imageshape):
    points1 = np.array(pcd1.points)
    points2 = np.array(pcd2.points)
    pointshelper = np.array(pcdhelper.points)
    print("matching ...")
    nbrs = NearestNeighbors(n_neighbors=1, n_jobs=10).fit(pointshelper)
    distance, indices = nbrs.kneighbors(points1)
    print("all point count:", len(distance))
    threshold = np.percentile(np.array(distance), 5)
    matchlist1 = []
    pointlist1 = []
    # print("---",indices)
    for i in range(len(distance)):
        if distance[i][0] <= threshold:
            pointlist1.append(points1[i])
            temp = decompose(indices[i][0], imageshape)
            temp.append(1)

            matchlist1.append(temp)

    distance, indices = nbrs.kneighbors(points2)
    threshold = np.percentile(np.array(distance), 5)
    matchlist2 = []
    pointlist2 = []

    for i in range(len(distance)):
        if distance[i][0] <= threshold:
            pointlist2.append(points2[i])
            temp = decompose(indices[i][0], imageshape)
            temp.append(1)
            matchlist2.append(temp)
    matchlist1 = np.array(matchlist1)
    matchlist2 = np.array(matchlist2)

    pointlist1 = np.array(pointlist1)
    pointlist2 = np.array(pointlist2)
    return matchlist1, pointlist1, matchlist2, pointlist2


Intrinsic = np.array([[1169.621094, 0.000000, 646.295044],
                      [0.000000, 1167.105103, 489.927032],
                      [0.000000, 0.000000, 1.000000]])
kx, ky = 640 / 1296, 480 / 968
Intrinsic[0] = Intrinsic[0] * kx
Intrinsic[1] = Intrinsic[1] * ky  # 大图+修正
intrinsic = np.array([[577.590698, 0, 318.905426, 0.000000],
                      [0, 578.729797, 242.683609, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])  # 深度图
##############################
pcd1 = get3dPointCloud(0)
print("read pointcloud 1")
pcd2 = get3dPointCloud(101)
print("read pointcloud 2")
pcdhelper = get3dPointCloud(50)
print("read pointcloud helper")

Image = getColor(50)
pose = readCameraMatrix("./pose/" + str(50) + ".txt")
pcd2.transform(pose)  ##################3
matchlist1, pointlist1, matchlist2, pointlist2 = correpondence(pcd1, pcd2, pcdhelper, imageshape=(480, 640))
opt = Optimizer(1e-4)
pose_tar = np.linalg.inv(pose)
print("pose verse,target", np.linalg.inv(pose))
i = 0
E0 = pose[:3,:]
E = []
for i in range(len(matchlist2)):
    E.append(E0.copy())
E = np.array(E)

for i in range(200):
    #import time
    #matchlist2 pointlist2
    #start = time.clock()
    P = np.array(pcd1.points)
    Q = np.array(pcd2.points)
    ##########sample
    #P = np.array(random.sample(list(P), 30000))
    #Q = np.array(random.sample(list(Q), 30000))
    ##########sample
    P, Q, indice = point_matching(P, Q)
    NOTC = np.zeros((1, 3))
    NOTQ = np.zeros((1, 3))
    loss = opt.loss(P, Q, Intrinsic, E, matchlist2, pointlist2)

    R = opt.pointcloud_R(P, Q, Intrinsic, E, matchlist2, pointlist2)
    Q = (R.dot(Q.T)).T
    pointlist2 = (R.dot(pointlist2.T)).T
    T = pack_Rt(R, np.zeros(3))
    pcd2.transform(T)
    t = opt.pointcloud_t(P, Q, Intrinsic, E, matchlist2, pointlist2)
    pointlist2 = pointlist2+t
    T = pack_Rt(np.eye(3), t)
    pcd2.transform(T)
    #end = time.clock()
    pose = pose.dot(pack_Rt(R, t))

    t_dist = np.linalg.norm(t)
    R_angle= np.arccos(np.clip((np.sum(np.diag(R)) - 1.0) / 2, -1, 1)) / np.pi * 180.0
    gt_angle = np.arccos(np.clip((np.sum(np.diag(pose[:3,:3])) - 1.0) / 2, -1, 1)) / np.pi * 180.0
    print("-----------------------")
    print("loss:", loss)
    print("R", R_angle)
    print("t", t_dist)
    print("gt_angle",gt_angle)
    #print("time cost =", end - start)
    # time.sleep(60)
    #if i%10 ==9:
    #    o3d.draw_geometries([pcd1, pcd2])
