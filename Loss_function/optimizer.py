#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2019/8/26 23:51
# @Author: csc
# @File  : optimizer.py
import numpy as np

##finished
def cross_op(x):
    u, v, w = x[0], x[1], x[2]
    cross_x_half = np.array([
        [0, -w, v],
        [0, 0, -u],
        [0, 0, 0]])
    return cross_x_half - (cross_x_half.T)


def Vec2Rot(r):
    theta = np.linalg.norm(r, 2)
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    R = np.cos(theta) * np.eye(3) + np.sin(theta) * cross_op(k) + (1 - np.cos(theta)) * np.outer(k, k)
    return R


class Optimizer(object):
    def __init__(self):
        '''
        object funtion:
        minmize sum ||P - T(Q)|| + ||C(I) x T(Q)|| :
        #never mind intrinsic matrix
        '''
        pass

    def simple(self,m,n):
        np.random.seed(10)
        P = np.random.random((m, 3))
        Q = np.random.random((m, 3))
        #P = np.zeros((1, 3))
        #Q = np.zeros((1, 3))
        c = np.random.random((n, 2))
        c = np.concatenate((c, np.ones((n, 1))), axis=1)
        R = np.random.random((3, 3))
        U, sigma, V = np.linalg.svd(R)
        R = U.dot(V)
        E0 = np.random.random((3, 4))
        E0[:3, :3] = R
        E =[]
        for i in range(n):
            E.append(E0.copy())
        Q_ = np.random.random((n, 3))
        return P,Q,E,c,Q_

    def pointcloud_R(self,P, Q, E, c, Q_):
        '''
        minmize sum ||P-R(Q)||2 + ||I x E R(Q')||2
        input: all should be numpy
        P matched(Q) points in pcd1             n*3
        Q matched(P) points in pcd2             n*3
        c matched(Q') pixel (I)                 m*3
        Q_ matched(I) point                     m*3
        camera pose matched(I)
        E extrinsic matrix                      m*3*4
        ---------------------------------------
        minmize sum ||P+Qxr||2+||-c x Re Q x r + c x Re Q + c x te||2  (R = I+rx)
        ==> 2 g r + r' H r
        '''
        n1 = np.shape(P)[0]
        g = np.zeros(3)
        n2 = np.shape(c)[0]
        H = np.zeros((3, 3))
        for i in range(n1):
            g += np.cross(P[i, :], Q[i, :])
            TEMPQ = np.array([Q[i,:]]).T.dot(np.array([Q[i,:]]))
            #print(TEMPQ.shape)
            H +=TEMPQ

        for i in range(n2):
            Re = E[i][:3, :3]
            te = np.array([E[i][:3, 3]]).T
            qi_ = np.array([Q_[i]]).T
            subg = cross_op(c[i]).dot(Re.dot(qi_) + te)

            #subg1 = cross_op(c[i]).dot(Re).dot(Q_[i].T)
            #subg2 = np.cross(c[i], te)
            #print(subg,subg1+subg2)

            h = -cross_op(c[i]).dot(Re).dot(cross_op(Q_[i]))
            H += h.T.dot(h)
            g += (subg).T.dot(h)[0]

        r = -np.linalg.solve(H, g)
        R = Vec2Rot(r)
        return R

    def pointcloud_t(self,P, Q, E, c, Q_):
        '''
        minmize sum ||P-T(Q)||2 + ||I x E T(Q')||2
        minmize sum ||P-Q-t||2+||cxRe(q+t)||2
        ==> gt+tHt
        '''
        n1 = np.shape(P)[0]
        n2 = np.shape(Q_)[0]
        H = np.zeros((3, 3))
        g = np.zeros(3)

        for i in range(n1):
            g += Q[i, :] - P[i, :]
            H += np.eye(3)

        for i in range(n2):
            Re = E[i][:3, :3]
            te = np.array([E[i][:3, 3]]).T
            qi_ = np.array([Q_[i]]).T
            subg1 = cross_op(c[i]).dot(Re).dot(qi_)
            subg2 = cross_op(c[i]).dot(te)
            subg = (subg1+subg2).T
            h = cross_op(c[i]).dot(Re)
            g += subg.dot(h)[0]
            H += h.T.dot(h)
        t = -np.linalg.solve(H, g)
        return t

    # I will change E at the same time
    # 这里的E只能是同一个

    def image_R(self,E, c, Q_,Intrinsic,test=0):##have some problem
        '''
        ||I x Enew Q'||2
        '''
        #print(E[0].shape)
        n2 = np.shape(Q_)[0]
        #print(n2,len(E))
        H = np.zeros((3, 3))
        g = np.zeros(3)

        for i in range(n2):
            E0 = Intrinsic.dot(E[i])
            Re = E0[:3, :3]
            te = np.array([E0[:3, 3]]).T
            qi_ = np.array([Q_[i]]).T
            subg = cross_op(c[i]).dot(Re.dot(qi_) + te)
            h = -cross_op(c[i]).dot(Re).dot(cross_op(Q_[i]))
            H += h.T.dot(h)
            g += (subg).T.dot(h)[0]

        r = -np.linalg.solve(H, g)
        R = Vec2Rot(r)
        #U,sig,V = np.linalg.svd(np.eye(3)+cross_op(r))
        #R = U.dot(V)
        delta_angle = np.arccos(np.clip((np.sum(np.diag(R)) - 1.0) / 2, -1, 1)) / np.pi * 180.0
        print("rotate:",delta_angle)

        for i in range(len(E)):
            Re = E[i][:3,:3]
            Re = Re.dot(R)
            E[i][:3,:3] = Re

        if test==1:
            return E,R
        return E

    # 这里的E只能是一个
    # I will change E at the same time
    def image_t(self, E, c, Q_,Intrinsic):
        '''
        ||subg + h t||
        '''
        n2 = np.shape(Q_)[0]
        H = np.zeros((3, 3))
        g = np.zeros(3)
        for i in range(n2):
            E0 = Intrinsic.dot(E[i])
            Re = E0[:3,:3]
            h = cross_op(c[i]).dot(Intrinsic)
            subg = cross_op(c[i]).dot(Re).dot(Q_[i])
            g += subg.T.dot(h)
            H += h.T.dot(h)
        t = -np.linalg.solve(H, g)
        move_distance = np.linalg.norm(t-E[0][:,3])**2
        print("image move distance ",move_distance)
        for i in range(n2):
            E[i][:, 3] = t
        return E

    def loss(self,P, Q, Intrinsic, E, c, Q_):
        loss_pointcloud = np.linalg.norm(P - Q, 'fro') ** 2
        n2 = np.shape(Q_)[0]
        loss_camera = 0
        for i in range(n2):
            ci = cross_op(c[i])
            qi_ = np.array([Q_[i]]).T
            E0 = Intrinsic.dot(E[i])
            Re = E0[:3, :3]
            te = np.array([E0[:3, 3]]).T
            l = ci.dot(Re.dot(qi_) + te)
            #print(l.shape)
            loss_camera += l.T.dot(l)
        loss = loss_pointcloud + loss_camera
        loss = loss/1e10
        #print("loss", loss_pointcloud,loss_camera)
        return loss

def opttest():
    opt = Optimizer()
    # P, Q, E, c, Q_ = opt.easysimple()
    P, Q, E, c, Q_ = opt.simple(1000, 300)

    for i in range(100):
        loss = opt.loss(P, Q, E, c, Q_)
        R = opt.pointcloud_R(P, Q, E, c, Q_)
        Q = (R.dot(Q.T)).T
        Q_ = (R.dot(Q_.T)).T
        delta_angle = np.arccos(np.clip((np.sum(np.diag(R)) - 1.0) / 2, -1, 1)) / np.pi * 180.0
        # delta_t =0
        # delta_angle = 0
        t = opt.pointcloud_t(P, Q, E, c, Q_)
        delta_t = np.linalg.norm(t, 2)
        Q += t[np.newaxis, :]
        Q_ += t[np.newaxis, :]
        # print(i,loss,delta_angle,delta_t)
        print('iter=%g, loss=%g, angle=%g, delta_t=%g' % (i, loss, delta_angle, delta_t))
        E = opt.image_R(E,c,Q)
        E = opt.image_t(E,c,Q)