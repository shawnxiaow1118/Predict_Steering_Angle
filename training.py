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

input_path = "angles_train.txt"
BATCH_SIZE = 50

def running():
	with tf.Graph().as_default():
		global_step = tf.Variable(0, trainable = False)

		## get input
		images, angles = input.input(input_path, BATCH_SIZE)

		## inference build model
		prediction = pred_steer.inference(images)

		## calculate loss
		with tf.name_scope('loss'):
			loss = pred_steer.loss(prediction, angles)
			tf.scalar_summary('loss', loss)

		## build model per batch and update parameters
		train_op = pred_steer.train(loss, 0.002, global_step)

		## build initialization peration 
		init = tf.initialize_all_variables()
		
		## merge all summaries and initialize writer
		#summary_op = tf.merge_all_summaries()
		#train_writer = tf.train.SummaryWriter("./tensorboard", graph = tf.get_default_graph())

		sess = tf.Session()

		sess.run(init)
		## start the queue runners
		coord = tf.train.Coordinator()
		enqueue_threads = tf.train.start_queue_runners(sess = sess, coord = coord)
		for step in range(0,5000):
			start_time = time.time()
			_, loss_value, pred = sess.run([train_op, loss, prediction])
			#_, summary = sess.run([train_op, summary_op])
			#train_writer.add_summary(summary, step)
			duration = time.time() - start_time
			print(str(step) + " time:"+ str(duration) + " loss: " + str(loss_value))
			print(pred)
		coord.request_stop()
		coord.join(enqueue_threads)

def main(argv = None):
	running()

if __name__=='__main__':
	tf.app.run()