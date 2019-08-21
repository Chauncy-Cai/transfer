#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/8/17 6:01
#@Author: csc
#@File  : main.py

import tensorflow as tf
import numpy as np
import cv2

#权重初始化函数
def weight_variable(shape, name=None):
    #输出服从截尾正态分布的随机值
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,name = name)

#偏置初始化函数
def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

#创建卷积op
#x 是一个4维张量，shape为[batch,height,width,channels]
#卷积核移动步长为1。填充类型为SAME,可以不丢弃任何像素点, VALID丢弃边缘像素点
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

#创建池化op
#采用最大池化，也就是取窗口中的最大值作为结果
#x 是一个4维张量，shape为[batch,height,width,channels]
#ksize表示pool窗口大小为2x2,也就是高2，宽2
#strides，表示在height和width维度上的步长都为2
def max_pool_2x2(x,name = None):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],
                          strides=[1,2,2,1], padding="SAME", name = name)
 
#a level of cnn
#weight and bias initialize is not included
#subname rule: levelnum_parameterindex
def cnn_level(x,W,b,subname:str, maxpool=False):
    #x is a tensor
    h_conv = tf.nn.relu(conv2d(x, W) + b, name = "h_conv" + subname)
    if maxpool:
        h_pool = max_pool_2x2(h_conv, name = "h_pool" + subname)
        return h_pool
    return h_conv
#a share level of cnn
#weight and bias will be share by two input
#weight and bias initialization is included
def cnn_level_share(x1,x2,cnn_shape,levelindex,maxpool = False):
    W = weight_variable(cnn_shape, name = "W_conv" + str(levelindex))
    b = bias_variable([cnn_shape[3]], name = "b_conv" + str(levelindex))
    re1 = cnn_level(x1, W, b, str(levelindex) + "_1", maxpool)
    re2 = cnn_level(x2, W, b, str(levelindex) + "_2", maxpool)
    return re1,re2


#a share level of fully connect
def fully_connect_share(x1,x2,inputshape,outputshape,levelindex):
    W = weight_variable([inputshape,outputshape],name = "W_fully_connect"+str(levelindex))
    b = bias_variable([outputshape],name = "b_fully_connect"+str(levelindex))

    re1 = tf.nn.relu(tf.matmul(x1,W)+b)
    re2 = tf.nn.relu(tf.matmul(x2,W)+b)
    return re1,re2

######################################################################
#generate three placehold
# x1,x2 is input image 31*31*3
# matched is vector showing whether x1,x2 match
x1 = tf.placeholder("float",shape=[None,256,256,3])
x2 = tf.placeholder("float",shape=[None,256,256,3])
matched = tf.placeholder("float",shape=[None])

## this model is VGG-11

#LAYER ONE 2-CONV3-64 + MAXPOOL (256*256*3=>128*128*64)
l11x1, l11x2 = cnn_level_share(x1, x2, [3,3,3,64], 1, maxpool=False)
l12x1, l12x2 = cnn_level_share(l11x1, l11x2, [3,3,64,64], 2, maxpool=True)

#LAYER TWO 2-CONV3-128 + MAXPOOL (128*128*64=>64*64*128)
l21x1, l21x2 = cnn_level_share(l12x1, l12x2, [3,3,64,128], 3, maxpool=False)
l22x1, l22x2 = cnn_level_share(l21x1, l21x2, [3,3,128,128], 4, maxpool=True)

#LAYER THREE 2-CONV3-256 + MAXPOOL (64*64*128=>32*32*256)
l31x1, l31x2 = cnn_level_share(l22x1, l22x2, [3,3,128,256], 5, maxpool=False)
l32x1, l32x2 = cnn_level_share(l31x1, l31x2, [3,3,256,256], 6, maxpool=True)
 
#LAYER FOUR 2-CONV3-512 + MAXPOOL (32*32*256=>8*8*512)
l41x1, l41x2 = cnn_level_share(l32x1, l32x2, [3,3,256,512], 7, maxpool=True)
l42x1, l42x2 = cnn_level_share(l41x1, l41x2, [3,3,512,512], 8, maxpool=True)

#LAYER SIX full-connected
flat1 = tf.reshape(l42x1,[-1,8*8*512])
flat2 = tf.reshape(l42x2,[-1,8*8*512])

l61x1, l61x2 = fully_connect_share(flat1,flat2,8*8*512,4096,11)
l62x1, l62x2 = fully_connect_share(l61x1,l61x2,4096,1024,12)
l63x1, l63x2 = fully_connect_share(l62x1,l62x2,1024,256,13)
l64x1, l64x2 = fully_connect_share(l63x1,l63x2,256,32,14)
#------ soft max-----

#FINAL
#normalize
y1 = tf.nn.l2_normalize(l64x1,dim=1)##?all?
y2 = tf.nn.l2_normalize(l64x2,dim=1)##?all? [?,30]
#loss
judge = tf.reduce_sum(tf.multiply(y1,y2),1)
loss_vec = tf.multiply(judge,matched)
loss = tf.reduce_sum(tf.negative(loss_vec))

#train_step
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
#correct_predict
correct_predict = tf.greater(loss_vec,0.5)
accuracy = tf.reduce_mean(tf.cast(correct_predict,"float"))

#####calculate F1 SCORE
possign = (matched+1)/2
negsign = (-matched+1)/2

positivelen = tf.reduce_sum(possign)
negativelen = tf.reduce_sum(negsign)

correctcast = tf.cast(correct_predict,"float")
TP = tf.reduce_sum(tf.multiply(correctcast,possign)) ##TP
TN = tf.reduce_sum(tf.multiply(correctcast,negsign)) ##TN
FN = positivelen - TP ## FN
FP = negativelen - TN ## FP


epsilon = 1e-7

precision =TP / (TP + FP + epsilon) ##TP/(TP+FP)
recall =TP / (TP + FN + epsilon)

F1 =2 * (precision * recall) / (precision + recall)



def readImages():
    truthbase = "./box/true_image/"
    faultbase = "./box/wrong_image/"
    x1=[]
    x2=[]
    matched=[]
    print("data loading...")
    for i in range(308):
        index = i
        x1.append(cv2.imread(truthbase + str(index) + "_1.jpg"))
        x2.append(cv2.imread(truthbase + str(index) + "_2.jpg"))
        matched.append(1)
        x1.append(cv2.imread(faultbase + str(index) + "_1.jpg"))
        x2.append(cv2.imread(faultbase + str(index) + "_2.jpg"))
        matched.append(-1)
    print("x1:",np.array(x1).shape," x2:",np.array(x2).shape," matched:",len(matched))
    return [x1,x2,matched]

def cnn_train(epo):
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    data = readImages()
    for i in range(epo):
        start = (i % 13) * 25
        end = start + 25
        batch = [data[0][start:end],
                 data[1][start:end],
                 data[2][start:end]]##format [x1,x2,match?]
        if i==0:
            print("standard batch:", np.array(batch[0]).shape)
            print("batch situation:",len(batch[0]), len(batch[1]), len(batch[2]))
        if i%10 ==0:
            train_accuracy,train_loss,train_precision,train_recall,train_F1,ZTP,ZTN =  sess.run(
                    [accuracy,loss,precision,recall,F1,TP,TN],feed_dict={x1:data[0],x2:data[1],matched:data[2]
            })
            print ("step %d, training accuracy %g, loss %g, precision %g, recall %g, F1 %g,TP %g,TN %g"
                    % (i, train_accuracy, train_loss, train_precision, train_recall, train_F1,ZTP,ZTN))
        train_step.run(feed_dict={x1:batch[0],x2:batch[1],matched:batch[2]})

def predict():
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    #test_accuracy = accuracy.eval(feed_dict={
    #        x1:batch[0],x2:batch[1],matched:batch[2]})
    #print("test accuracy %g"% test_accuracy)


cnn_train(600)
#predict()
