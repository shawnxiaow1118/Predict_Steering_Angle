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
import cv2
import time
import pandas as pd
import time

## image number to predict one step
BATCH_SIZE = 1

## test image path
# test_path = "./test_data/test_list.txt"
## or just read the images folder

path = "./test_data/center/"
test_file_name = os.listdir(path)

##test_file_name = read_data.read_test(dirs)

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
saver.restore(sess, "./save/my-model-189000")

predictions = []
names  = []
for name in test_file_name:
	start = time.time()
	#img = cv2.imread('./test_data/center/'+name)
	image = read_data.read_image("./test_data/center/"+name)
	pred = sess.run(prediction, feed_dict = {x: [image], train_flag: False, drop_prob:1.0, wd:0.0})
	#cv2.imshow('frame',img)
	#cv2.waitKey(1)
	#time.sleep(0.5)
	#print(pred[0][0])
	duration = time.time()-start
	print("prediction for image: " + name+ " is " +str(pred)+" time: "+str(duration))
	names.append(str(name[0:19]))
	predictions.append(pred[0][0])

out = pd.DataFrame({'frame_id': names,'steering_angle': predictions})

out.to_csv('submission.csv', index = False)

