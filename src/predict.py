#-*- coding: utf-8 -*-
# Predict for test data using saved model and ensemble methods
#
# *** Udacity Predict steering angle ****
#
# Copyright 2016 Zhaocheng Liu, Xiao Wang
# {zcliu,  xwang696}@gatech.edu

import tensorflow as tf
import pred_steer
import read_data
import time
import os

## image number to predict one step
BATCH_SIZE = 1

## test image path
test_path = "./data/test.txt"


test_file_name = read_data.read_test(test_path)

init = tf.initialize_all_variables()
sess = tf.InteractiveSession()

# Launch the graph
x = tf.placeholder("float", shape=(BATCH_SIZE,140,320,3),name='x_all')
drop_prob = tf.placeholder('float', name='drop_prob')
wd = tf.placeholder('float', name='wd')
train_flag = tf.placeholder(tf.bool, name='train_flag')
sess.run(init)
prediction = pred_steer.inference(x, train_flag, drop_prob, wd)
saver = tf.train.Saver()
saver.restore(sess, "./save/model.ckpt")

for name in test_file_name:
	image = read_data.read_image("./data/images/"+name)
	pred = sess.run(prediction, feed_dict = {x: [image], train_flag: False, drop_prob:1.0, wd:0.0})
	print(pred)