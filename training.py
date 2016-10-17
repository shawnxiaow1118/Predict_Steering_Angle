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
eval_path = "angles_valid.txt"
BATCH_SIZE = 50

def running():
	with tf.Graph().as_default():
		global_step = tf.Variable(0, trainable = False)

		## get input
		images, angles = input.input(input_path, BATCH_SIZE)

		## inference build model
		prediction = pred_steer.inference(images)

		## calculate loss
		loss = pred_steer.loss(prediction, angles)

		## build model per batch and update parameters
		train_op = pred_steer.train(loss, 0.0008, global_step)

		## get evaluation set 
		eval_imgs, eval_angs = input.input(eval_path, BATCH_SIZE)

		## evaluation prediction
		eval_pred = pred_steer.inference(eval_imgs)

		## calculate evaluation loss
		eval_loss = pred_steer.loss(eval_pred, eval_angs)

		## build initialization peration 
		init = tf.initialize_all_variables()
		## merge all summaries and initialize writer
		#summary_op = tf.merge_all_summaries()
		#train_writer = tf.train.SummaryWriter("./tensorboard", graph = tf.get_default_graph())

		tf.scalar_summary('train_RMSE', loss)
		#tf.scalar_summary('train_pred', tf.reduce_mean(prediction))
		#tf.scalar_summary('eval_pred', tf.reduce_mean(eval_pred))
		#tf.scalar_summary('train_angle', tf.reduce_mean(tf.string_to_number(angles, out_type = tf.float32)))
		#tf.scalar_summary('eval_angle', tf.reduce_mean(tf.string_to_number(eval_angs, out_type = tf.float32)))

		sess = tf.Session()
		merged = tf.merge_all_summaries()
		writer = tf.train.SummaryWriter("./tensor/", sess.graph)
		saver = tf.train.Saver()


		sess.run(init)
		## start the queue runners
		coord = tf.train.Coordinator()
		enqueue_threads = tf.train.start_queue_runners(sess = sess, coord = coord)
		for step in range(1,60000):
			start_time = time.time()
			_, summary = sess.run([train_op, merged])
			if step%10 == 0:
				print("step %d, val loss %g"%(step, sess.run(eval_loss)))
			if step%100 == 0:
				checkpath = "./save/model.ckpt"
				filename = saver.save(sess, checkpath)
				print("Model saved in file: %s" %filename)
			#_, summary = sess.run([train_op, summary_op])
			#train_writer.add_summary(summary, step)
			duration = time.time() - start_time
			writer.add_summary(summary, step)
			print(str(step) + " time:"+ str(duration))# + " loss: " + str(loss_value))
			#print(pred)
		coord.request_stop()
		coord.join(enqueue_threads)

def main(argv = None):
	running()

if __name__=='__main__':
	tf.app.run()