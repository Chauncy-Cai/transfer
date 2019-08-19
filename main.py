#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/8/17 6:01
#@Author: csc
#@File  : main.py

import tensorflow as tf
import numpy as np
import cv2


#权重初始化函数
def weight_variable(shape):
    #输出服从截尾正态分布的随机值
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

#偏置初始化函数
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

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
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],
                          strides=[1,2,2,1], padding="SAME")

######################################################################
#generate three placehold
# x1,x2 is input image 31*31*3
# matched is vector showing whether x1,x2 match
x1 = tf.placeholder("float",shape=[None,256,256,3])
x2 = tf.placeholder("float",shape=[None,256,256,3])
matched = tf.placeholder("float",shape=[None])
#LAYER ONE cnn
W_conv1 = weight_variable([5,5,3,10])
b_conv1 = bias_variable([10])

h_conv1_1 = tf.nn.relu(conv2d(x1,W_conv1)+b_conv1)
h_pool1_1 = max_pool_2x2(h_conv1_1)
h_conv1_2 = tf.nn.relu(conv2d(x2,W_conv1)+b_conv1)
h_pool1_2 = max_pool_2x2(h_conv1_2)
#output shape here should be [batch,128,128,10]

#LAYER TWO cnn
W_conv2 = weight_variable([5,5,10,16])
b_conv2 = weight_variable([16])

h_conv2_1 = tf.nn.relu(conv2d(h_pool1_1, W_conv2) + b_conv2)
h_pool2_1 = max_pool_2x2(h_conv2_1)
h_conv2_2 = tf.nn.relu(conv2d(h_pool1_2, W_conv2) + b_conv2)
h_pool2_2 = max_pool_2x2(h_conv2_2)
#output shape here should be [batch,64,64,16]

#LAYER THREE CNN
W_conv3 = weight_variable([5,5,16,32])
b_conv3 = weight_variable([32])

h_conv3_1 = tf.nn.relu(conv2d(h_pool2_1, W_conv3) + b_conv3)
h_pool3_1 = max_pool_2x2(h_conv3_1)
h_conv3_2 = tf.nn.relu(conv2d(h_pool2_2, W_conv3) + b_conv3)
h_pool3_2 = max_pool_2x2(h_conv3_2)
#output shape here should be [batch,32,32,32]

#LAYER FOUR full-connected
h_pool2_flat_1 = tf.reshape(h_pool3_1,[-1,32*32*32])
h_pool2_flat_2 = tf.reshape(h_pool3_2,[-1,32*32*32])

W_fc1 = weight_variable([32*32*32,120])
b_fc1 = bias_variable([120])

h_fc1_1 = tf.nn.relu(tf.matmul(h_pool2_flat_1,W_fc1)+b_fc1)
h_fc1_2 = tf.nn.relu(tf.matmul(h_pool2_flat_2,W_fc1)+b_fc1)

#LAYER FOUR full-connected
W_fc2 = weight_variable([120,30])
b_fc2 = bias_variable([30])

h_fc2_1 = tf.nn.relu(tf.matmul(h_fc1_1,W_fc2)+b_fc2)
h_fc2_2 = tf.nn.relu(tf.matmul(h_fc1_2,W_fc2)+b_fc2)

#FINAL
#normalize
y1 = tf.nn.l2_normalize(h_fc2_1,dim=1)##?all?
y2 = tf.nn.l2_normalize(h_fc2_2,dim=1)##?all? [?,30]
#loss
loss_vec = tf.multiply(tf.reduce_sum(tf.multiply(y1, y2),1),matched)
loss = tf.reduce_sum(tf.negative(loss_vec))


#train_step
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
#correct_predict
correct_predict = tf.greater(loss_vec,0.8)
accuracy = tf.reduce_mean(tf.cast(correct_predict,"float"))

def readImages():
    truthbase = "E:/research\Austin-project\coding/transfer/box/true_image/"
    faultbase = "E:/research/Austin-project/coding/transfer/box/wrong_image/"
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
            print("standard batch:",np.array(batch[0]).shape)
        if i%10 ==0:
            train_accuracy,train_loss =  sess.run([accuracy,loss],feed_dict={
                x1:batch[0],x2:batch[1],matched:batch[2]
            })
            print ("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x1:batch[0],x2:batch[1],matched:batch[2]})

def predict():
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    #test_accuracy = accuracy.eval(feed_dict={
    #        x1:batch[0],x2:batch[1],matched:batch[2]})
    #print("test accuracy %g"% test_accuracy)


cnn_train(2000)
#predict()