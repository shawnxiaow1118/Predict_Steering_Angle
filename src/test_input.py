#-*- coding: utf-8 -*-
# Read data and feeddict
#
# *** Udacity Predict steering angle ****
#
# Copyright 2016 Zhaocheng Liu, Xiao Wang
# {zcliu,  xwang696}@gatech.edu

import tensorflow as tf
import input

init_op = tf.initialize_all_variables()
images, angles = input.distorted_inputs("../data/angles_train.txt", 10)
eval_i, eval_a = input.origin_inputs("../data/angles_valid.txt", 10)
#images,angles = input.input("./image/angles.txt", 3)

#with tf.name_scope('input'):
#	tf.image_summary('input', images, max_images = 2)

#merged = tf.merge_all_summaries()
#input_writer = tf.train.SummaryWriter('./', graph = tf.get_default_graph())

with tf.Session() as sess:
	sess.run(init_op)
	sess.run(tf.initialize_local_variables())
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord, sess = sess)
	for i in range(0,100):
		#im = sess.run(images)
		an = sess.run(angles)
		an_e = sess.run(eval_a)
		#summary = sess.run(merged)
		#input_writer.add_summary(summary,i)
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
		print(an)
		print("====")
		print(an_e)
		print("++++++++")
	coord.request_stop()
	coord.join(threads)

