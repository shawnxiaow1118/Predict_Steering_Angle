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
import os

folder = './tensor/'
for the_file in os.listdir(folder):
	file_path = os.path.join(folder, the_file)
	os.remove(file_path)

input_path = "./data/angles_train.txt"
eval_path = "./data/angles_valid.txt"

### indicate whether in training phase or in testing case, used for batch normalization
train_flag = tf.placeholder(tf.bool, name='train_flag')
### actually this is the keep probability in those fully connected layers
drop_prob = tf.placeholder('float', name='drop_prob')
wd = tf.placeholder('float', name='wd')

def running(learning_rate, keep_prob, BATCH_SIZE, weight_decay):
	x = tf.placeholder(tf.float32, [BATCH_SIZE, 480, 640, 3])
	y = tf.placeholder(tf.float32, [BATCH_SIZE])

	global_step = tf.Variable(0, trainable = False)

	##### training queue inputs #####

	## get input
	images, angles = input.distorted_inputs(input_path, BATCH_SIZE)

	## inference build model
	prediction = pred_steer.inference(x, train_flag, drop_prob, wd)

	## calculate loss
	loss = pred_steer.loss(prediction, y)

	## build model per batch and update parameters
	train_op = pred_steer.train(loss, learning_rate, global_step)

	## get evaluation set 
	eval_imgs, eval_angs = input.origin_inputs(eval_path, BATCH_SIZE)

	## build initialization peration 
	init = tf.initialize_all_variables()
		## merge all summaries and initialize writer
		#summary_op = tf.merge_all_summaries()
		#train_writer = tf.train.SummaryWriter("./tensorboard", graph = tf.get_default_graph())

	tf.scalar_summary('train_RMSE', loss)
		#tf.scalar_summary('train_pred', tf.reduce_mean(prediction))
		#tf.scalar_summary('eval_pred', tf.reduce_mean(eval_pred))
		#tf.scalar_summary('train_angle', tf.reduce_mean(angles))
		#tf.scalar_summary('eval_angle', tf.reduce_mean(tf.string_to_number(eval_angs, out_type = tf.float32)))

	sess = tf.Session()
	merged = tf.merge_all_summaries()
	writer = tf.train.SummaryWriter("./tensor/", sess.graph)
	saver = tf.train.Saver()


	sess.run(init)
		## start the queue runners
	coord = tf.train.Coordinator()
	enqueue_threads = tf.train.start_queue_runners(sess = sess, coord = coord)
	for step in range(1,220000):
		start_time = time.time()
		images_array, angles_array = sess.run([images, angles])
		_, summary,pred1 = sess.run([train_op, merged, prediction], 
			feed_dict = {x: images_array, y: angles_array,train_flag:True, drop_prob:keep_prob, wd:weight_decay })
		if step%10 == 0:
			eval_images_array, eval_angles_array = sess.run([eval_imgs, eval_angs])
			#print("step: %d, eval_loss: %g"%(step, sess.run(loss, feed_dict = {
			#	x: eval_images_array, y:eval_angles_array, train_flag:False, drop_prob:1.0})))
			out = sess.run(loss, feed_dict = {x: eval_images_array, y:eval_angles_array, train_flag:False, drop_prob:1.0, wd:weight_decay})
			print("loss:" + str(out))
			# if step%100 == 0:
			# 	checkpath = "./save/model.ckpt"
			# 	filename = saver.save(sess, checkpath)
			# 	print("Model saved in file: %s" %filename)
			# _, summary = sess.run([train_op, summary_op])
			# train_writer.add_summary(summary, step)
		duration = time.time() - start_time
		writer.add_summary(summary, step)
		print(str(step) + " time:"+ str(duration))# + " loss: " + str(loss_value))
			#print(pred)
	coord.request_stop()
	coord.join(enqueue_threads)

def main(argv = None):
	## argv[4] = {name_of_py_file, learning_rate, drop_prob, BATCH_SIZE}
	running(float(argv[1]), float(argv[2]), int(argv[3]), float(argv[4]))

if __name__=='__main__':
	tf.app.run()