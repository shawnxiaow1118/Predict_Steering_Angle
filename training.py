#-*- coding: utf-8 -*-
# Model for predict steering
#
# *** Udacity Predict steering angle ****
#
# Copyright 2016 Xiao Wang, Zhaocheng Liu
# {xwang696, zcliu}@gatech.edu

import tensorflow as tf
import pred_steer
import input
import time

def running():
	with tf.Graph().as_default():
		global_step = tf.Variable(0, trainable = False)

		## get input
		images, angles = input.input("./image/angles.txt", 10)

		## inference build model
		prediction = pred_steer.inference(images)

		## calculate loss
		loss = pred_steer.loss(prediction, angles)

		## build model per batch and update parameters
		train_op = pred_steer.train(loss, 0.01, global_step)

		## build initialization peration 
		init = tf.initialize_all_variables()
		sess = tf.Session()

		sess.run(init)
		## start the queue runners
		coord = tf.train.Coordinator()
		enqueue_threads = tf.train.start_queue_runners(sess = sess, coord = coord)
		for step in range(0,10):
			start_time = time.time()
			_, loss_value = sess.run([train_op, loss])
			duration = time.time() - start_time
			print(loss_value)
		coord.request_stop()
		coord.join(enqueue_threads)

def main(argv = None):
	running()

if __name__=='__main__':
	tf.app.run()