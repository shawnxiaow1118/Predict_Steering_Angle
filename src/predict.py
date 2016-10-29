#-*- coding: utf-8 -*-
# Predict for test data using saved model and ensemble methods
#
# *** Udacity Predict steering angle ****
#
# Copyright 2016 Zhaocheng Liu, Xiao Wang
# {zcliu,  xwang696}@gatech.edu

import tensorflow as tf
import pred_steer
import input
import time
import os

## image number to predict one step
BATCH_SIZE = 1

## test image path
test_path = "angles_train.txt"


init = tf.initialize_all_variables()
sess = tf.InteractiveSession()

# Launch the graph
x = tf.placeholder("float", shape=(None,130),name='x_all')
train_flag = tf.placeholder(tf.bool, name='train_flag')
prediction = inference(x, train_flag)
saver = tf.train.Saver()
saver.restore(sess, "./model/my-model-111999")

sess.run(init)
batch_x,batch_y = LoadTrainBatch(batch_size)
pred = sess.run(prediction, feed_dict = {x: batch_x, train_flag: False})