#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/8/27 4:17
# @Author: csc
# @File  : main_img.py

from sklearn.neighbors import NearestNeighbors
from optimizer import *
from support import *

def correpondence(pcd1, pcd2, pcdhelper, imageshape):
    points1 = np.array(pcd1.points)
    points2 = np.array(pcd2.points)
    pointshelper = np.array(pcdhelper.points)
    print("matching ...")
    nbrs = NearestNeighbors(n_neighbors=1, n_jobs=10).fit(pointshelper)
    distance, indices = nbrs.kneighbors(points1)
    print("all point count:",len(distance))
    threshold = np.percentile(np.array(distance), 0.5)
    matchlist1 = []
    pointlist1 = []
    # print("---",indices)
    for i in range(len(distance)):
        if distance[i][0] <= threshold:
            pointlist1.append(points1[i])
            temp = decompose(indices[i][0], imageshape)
            temp.append(1)

            matchlist1.append(temp)

    #distance, indices = nbrs.kneighbors(points2)
    #threshold = np.percentile(np.array(distance), 20)
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

Intrinsic =np.array([[1169.621094,0.000000 ,646.295044],
                    [0.000000, 1167.105103, 489.927032],
                    [0.000000, 0.000000, 1.000000]])
kx, ky =640/1296,480/968
Intrinsic[0] = Intrinsic[0]*kx
Intrinsic[1] = Intrinsic[1]*ky#大图+修正
intrinsic = np.array([[577.590698, 0, 318.905426, 0.000000],
                     [0, 578.729797, 242.683609, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])#深度图
##############################
pcd1 = get3dPointCloud(0)
print("read pointcloud 1")
pcd2 = get3dPointCloud(101)
print("read pointcloud 2")
pcdhelper = get3dPointCloud(50)
print("read pointcloud helper")
Image = getColor(50)
pose = readCameraMatrix("./pose/" + str(50) + ".txt")

####################################################################
matchlist1, pointlist1, matchlist2, pointlist2 = correpondence(pcd1, pcd2, pcdhelper, imageshape=(480, 640))
###################################################################
opt = Optimizer()
# o3d.draw_geometries([pcd1,pcd2,pcdhelper,arrow])
pose_tar = np.linalg.inv(pose)
print("pose verse,target",np.linalg.inv(pose))
INDEX = 0
pose = np.eye(4)
####random position
A = np.random.random((3,3))
U,sig,V =  np.linalg.svd(A)
R = U.dot(V)
####
pos = 0
neg = 0


E0 = pose_tar[:3, :].copy()

for w in range(20):
    print("times:",w)
    np.random.seed(0)
    E0[:,3] = E0[:,3] + np.random.random(3)*10
    epo = 0
    while(1):
        #print("------"+str(epo)+"--------")
        epo += 1
        E = []
        for i in range(len(matchlist1)):
            E.append(E0.copy())
        # print("***",matchlist1,pointlist1)

        E,R = opt.image_R(E, matchlist1, pointlist1, Intrinsic,1)
        E = opt.image_t(E, matchlist1, pointlist1, Intrinsic)

        E0 = E[0].copy()

        ################
        exist_pose = E[0][:3,:3]
        delta_angle_gt = np.arccos(np.clip((np.sum(np.diag(exist_pose.dot(
            np.linalg.inv(pose_tar[:3,:3])
        ))) - 1.0) / 2, -1, 1)) / np.pi * 180.0
        distance_gt = np.linalg.norm(pose_tar[:3,3]-E[0][:3,3])
        end_target = np.arccos(np.clip((np.sum(np.diag(R)) - 1.0) / 2, -1, 1)) / np.pi * 180.0
        loss = opt.loss(np.zeros((1, 3)), np.zeros((1, 3)), Intrinsic, E, matchlist1, pointlist1)[0][0]
        if epo%10==0:
            print("angle:", delta_angle_gt, "distance", distance_gt,
                  "end_target:",end_target)
            print("--loss:",loss)
        if (epo>300) | (end_target<0.01) | (loss<1):
            if delta_angle_gt<5:
                pos += 1
            else:
                neg +=1
            break
print("possiblity",pos/(pos+neg))
    ###############
    #vis.update_geometry()

'''
def custom_draw_geometry_with_view_tracking(meshes):
    def track_view(vis):
        global matchlist1, pointlist1, INDEX, opt, arrow2, E0, Intrinsic, pose_tar
        if (INDEX % 1000 == 0):
            print("--------------")
            E = []
            for i in range(len(matchlist1)):
                E.append(E0.copy())
            # print("***",matchlist1,pointlist1)
            print("interation", str(INDEX / 1000), "--loss:",
                  opt.loss(np.zeros((1, 3)), np.zeros((1, 3)), Intrinsic,E, matchlist1, pointlist1)[0][0])
            E = opt.image_R(E, matchlist1, pointlist1,Intrinsic)
            E = opt.image_t(E, matchlist1, pointlist1,Intrinsic)

            E0 = E[0].copy()

            ################
            exist_pose = E[0][:3,:3]
            delta_angle_gt = np.arccos(np.clip((np.sum(np.diag(exist_pose.dot(
                np.linalg.inv(pose_tar[:3,:3])
            ))) - 1.0) / 2, -1, 1)) / np.pi * 180.0
            distance_gt = np.linalg.norm(pose_tar[:3,3]-E[0][:3,3])
            print("angel and distance 2 ground truth: angle:",delta_angle_gt,"distance",distance_gt)
            ###############
            vis.update_geometry()

        INDEX += 1

    o3d.draw_geometries_with_animation_callback(meshes, track_view)


custom_draw_geometry_with_view_tracking([pcd2, pcd1, pcdhelper])
'''
