#-*- coding: utf-8 -*-
# Read data and feeddict
#
# *** Udacity Predict steering angle ****
#
# Copyright 2016 Zhaocheng Liu, Xiao Wang
# {zcliu,  xwang696}@gatech.edu

import tensorflow as tf
import input

images_placeholder = tf.placeholder(tf.float32, shape=(12,480,640,3))

init_op = tf.initialize_all_variables()
images = input.distorted_inputs("./image/files.txt", 3)
#images = input.input("./image/files.txt", 3)

with tf.Session() as sess:
	sess.run(init_op)
	sess.run(tf.initialize_local_variables())
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord, sess = sess)
	for i in range(0,10):
		im = sess.run(images)
	# try:
	# 	while not coord.should_stop():
 #        	# Run training steps or whatever
	# 		im = sess.run(images)
	# 		print(im.shape)
	# 		ima = tf.gather(im, 0)
	# 		res = tf.image.decode_jpeg(ima)
	# 		print(res)
	# except tf.errors.OutOfRangeError:
	# 	print('Done training -- epoch limit reached')
	# finally:
 #    	# When done, ask the threads to stop.
	# 	coord.request_stop()
		print(im.shape)
	coord.request_stop()
	coord.join(threads)

