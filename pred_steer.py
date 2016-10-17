#-*- coding: utf-8 -*-
# Model for predict steering
#
# *** Udacity Predict steering angle ****
#
# Copyright 2016 Xiao Wang, Zhaocheng Liu
# {xwang696, zcliu}@gatech.edu

import tensorflow as tf
import math


def _variable_withweight_decay(name, shape, stddev, wd):
	""" Helper function to create an initialized Varibale with weight decay
	Iuput:
		name: the name of the variable
		shape: the shape of the variable, list of ints
		stddev: stadard deviation of a truncated Gaussian
		we: add L2 Loss weight decay numtiplied by this float. If None, weight 
		decay is 

	"""

def inference(images):
	""" 
	Input: batch images,
	Output: steering angle
	"""
	batch_size = images.get_shape()[0].value

	# with tf.variable_scope('conv1') as scope:
	# 	weights = tf.Variable(
	# 		tf.random_uniform([5,5,3,24], minval = -4*math.sqrt(6.0/(3.0+24.0)), maxval = , name='weights'))
	# 	biases = tf.Variable(tf.zeros([24]),
 #                         name='biases')

	# 	conv = tf.nn.conv2d(images, weights,[1,2,2,1],padding = 'VALID')
	# 	bias = tf.nn.bias_add(conv, biases)
	# 	conv_non = tf.sigmoid(bias, name = scope.name)

	with tf.variable_scope('conv1') as scope:
		weights = tf.Variable(
			tf.truncated_normal([5,5,3,12], stddev = math.sqrt(2.0/(25*3)), name='weights'))
		biases = tf.Variable(tf.zeros([12]),
                         name='biases')

		conv = tf.nn.conv2d(images, weights,[1,2,2,1],padding = 'VALID')
		bias = tf.nn.bias_add(conv, biases)
		conv_non = tf.nn.relu(bias, name = scope.name)


	with tf.variable_scope('pool1') as scope:
		pool = tf.nn.max_pool(conv_non, [1,3,3,1], [1,2,2,1], padding = 'VALID', name='pool1')

	with tf.variable_scope('conv2') as scope:
		weights = tf.Variable(
			tf.truncated_normal([5,5,12,24], stddev = math.sqrt(2.0/(25*12)), name='weights'))
		biases = tf.Variable(tf.zeros([24]),
                         name='biases')

		conv = tf.nn.conv2d(pool, weights,[1,2,2,1],padding = 'VALID')
		bias = tf.nn.bias_add(conv, biases)
		conv_non = tf.nn.relu(bias, name = scope.name)


	with tf.variable_scope('pool2') as scope:
		pool = tf.nn.max_pool(conv_non, [1,3,3,1], [1,2,2,1], padding = 'VALID', name='pool2')


	with tf.variable_scope('conv3') as scope:
		weights = tf.Variable(
			tf.truncated_normal([5,5,24,48], stddev = math.sqrt(2.0/(25*48)), name='weights'))
		biases = tf.Variable(tf.zeros([48]),
                         name='biases')

		conv = tf.nn.conv2d(pool, weights,[1,2,2,1],padding = 'VALID')
		bias = tf.nn.bias_add(conv, biases)
		conv_non = tf.nn.relu(bias, name = scope.name)


	with tf.variable_scope('pool3') as scope:
		pool = tf.nn.max_pool(conv_non, [1,3,3,1], [1,2,2,1], padding = 'VALID', name='pool3')


	with tf.variable_scope('local4') as scope:
		reshape = tf.reshape(pool,[batch_size,-1])
		dim = reshape.get_shape()[1].value
		weights = tf.Variable(
			tf.truncated_normal([dim, 100], stddev = math.sqrt(2.0/dim), name='weights'))
		biases = tf.Variable(tf.zeros([100]),
                         name='biases')
		local = tf.nn.relu(tf.matmul(reshape, weights)+biases, name = scope.name)

	with tf.variable_scope('local5') as scope:
		weights = tf.Variable(
			tf.truncated_normal([100, 25], stddev = math.sqrt(2.0/100), name='weights'))
		biases = tf.Variable(tf.zeros([25]),
                         name='biases')
		local = tf.nn.relu(tf.matmul(local, weights)+biases, name = scope.name)

	with tf.variable_scope('linear_regression') as scope:
		weights = tf.Variable(
			tf.truncated_normal([25, 1], stddev = math.sqrt(2.0/25), name='weights'))
		biases = tf.Variable(tf.zeros([1]),
                         name='biases')
		output = tf.matmul(local, weights) + biases
	return output

	# with tf.variable_scope('conv1') as scope:
	# 	weights = tf.Variable(
	# 		tf.truncated_normal([5,5,3,24], stddev = 5e-3, name='weights'))
	# 	biases = tf.Variable(tf.zeros([24]),
 #                         name='biases')

	# 	conv = tf.nn.conv2d(images, weights,[1,2,2,1],padding = 'VALID')
	# 	bias = tf.nn.bias_add(conv, biases)
	# 	conv_non = tf.sigmoid(bias, name = scope.name)


	# with tf.variable_scope('conv2') as scope:
	# 	weights = tf.Variable(
	# 		tf.truncated_normal([5,5,24,36], stddev = 5e-3, name='weights'))
	# 	biases = tf.Variable(tf.zeros([36]),
 #                         name='biases')

	# 	conv = tf.nn.conv2d(conv_non, weights,[1,2,2,1],padding = 'VALID')
	# 	bias = tf.nn.bias_add(conv, biases)
	# 	conv_non = tf.sigmoid(bias, name = scope.name)

	# with tf.variable_scope('conv3') as scope:
	# 	weights = tf.Variable(
	# 		tf.truncated_normal([5,5,36,48], stddev = 5e-3, name='weights'))
	# 	biases = tf.Variable(tf.zeros([48]),
 #                         name='biases')

	# 	conv = tf.nn.conv2d(conv_non, weights,[1,2,2,1],padding = 'VALID')
	# 	bias = tf.nn.bias_add(conv, biases)
	# 	conv_non = tf.sigmoid(bias, name = scope.name)

	# with tf.variable_scope('conv4') as scope:
	# 	weights = tf.Variable(
	# 		tf.truncated_normal([5,5,48,64], stddev = 5e-3, name='weights'))
	# 	biases = tf.Variable(tf.zeros([64]),
 #                         name='biases')

	# 	conv = tf.nn.conv2d(conv_non, weights,[1,2,2,1],padding = 'VALID')
	# 	bias = tf.nn.bias_add(conv, biases)
	# 	conv_non = tf.sigmoid(bias, name = scope.name)

	# with tf.variable_scope('conv5') as scope:
	# 	weights = tf.Variable(
	# 		tf.truncated_normal([5,5,64,64], stddev = 5e-3, name='weights'))
	# 	biases = tf.Variable(tf.zeros([64]),
 #                         name='biases')

	# 	conv = tf.nn.conv2d(conv_non, weights,[1,2,2,1],padding = 'VALID')
	# 	bias = tf.nn.bias_add(conv, biases)
	# 	conv_non = tf.sigmoid(bias, name = scope.name)

	# with tf.variable_scope('conv6') as scope:
	# 	weights = tf.Variable(
	# 		tf.truncated_normal([5,5,64,64], stddev = 5e-3, name='weights'))
	# 	biases = tf.Variable(tf.zeros([64]),
 #                         name='biases')

	# 	conv = tf.nn.conv2d(conv_non, weights,[1,2,2,1],padding = 'VALID')
	# 	bias = tf.nn.bias_add(conv, biases)
	# 	conv_non = tf.sigmoid(bias, name = scope.name)

	# with tf.variable_scope('local7') as scope:
	# 	reshape = tf.reshape(conv_non,[batch_size,-1])
	# 	dim = reshape.get_shape()[1].value
	# 	weights = tf.Variable(
	# 		tf.truncated_normal([dim, 100], stddev = 5e-6, name='weights'))
	# 	biases = tf.Variable(tf.zeros([100]),
 #                         name='biases')
	# 	local7 = tf.sigmoid(tf.matmul(reshape, weights)+biases, name = scope.name)



	# with tf.variable_scope('local8') as scope:
	# 	weights = tf.Variable(
	# 		tf.truncated_normal([100, 50], stddev = 5e-4, name='weights'))
	# 	biases = tf.Variable(tf.zeros([50]),
 #                         name='biases')
	# 	local8 = tf.sigmoid(tf.matmul(local7, weights)+biases, name = scope.name)

	# with tf.variable_scope('local9') as scope:
	# 	weights = tf.Variable(
	# 		tf.truncated_normal([50, 10], stddev = 5e-3, name='weights'))
	# 	biases = tf.Variable(tf.zeros([10]),
 #                         name='biases')
	# 	local9 = tf.sigmoid(tf.matmul(local8, weights)+biases, name = scope.name)

	# with tf.variable_scope('linear_regression') as scope:
	# 	weights = tf.Variable(
	# 		tf.truncated_normal([10, 1], stddev = 5e-2, name='weights'))
	# 	biases = tf.Variable(tf.zeros([1]),
 #                         name='biases')
	# 	output = tf.matmul(local9, weights) + biases
	# 	tf.scalar_summary("output", tf.reduce_mean(output))
	# return output, tf.get_variable("linear_regression.weights",[10,1])
		

def loss(output, angles):
	""" Calculate the RMSE of prediction angles and labeled angles
	Args:
		output:  prediction tensor, float [batch_size]
		angles:  label tesor, float [batch_size]

	Returns:
		loss: Loss tensor of type float
	"""
	batch_size = output.get_shape()[0].value
	
	reshape = tf.reshape(angles, [batch_size,1])

	#angs = tf.string_to_number(angles, out_type = tf.float32)
	angs = reshape
	### calculate square sum
	#angs = tf.sigmoid(angle)
	sum_square = tf.square(tf.sub(angs, output))
	### calculate mean square mean
	mean_square = tf.reduce_mean(sum_square)
	### take root

	#tf.scalar_summary('angle', tf.reduce_mean(angs))
	#tf.scalar_summary('pred', tf.reduce_mean(output))
	### do not take square root here! It has influence on the backprobagation
	loss = mean_square
	#tf.scalar_summary('RMSE', loss)
	return loss

def train(loss, learning_rate, global_step):
	""" Setting the training operation
	Args: 
		loss: loss tensor, from loss()
		learning_rate: the learning reate for gradient descent 
	"""
	### creat the optimizer using learing_rate
	optimizer = tf.train.AdamOptimizer(learning_rate)

	grads = optimizer.compute_gradients(loss)


	for var in tf.trainable_variables():
		tf.histogram_summary(var.op.name, var)

	for grad, var in grads:
		if grad is not None:
			tf.histogram_summary(var.op.name+"/gradients",grad)
	###  creat a global variable
	# global_step = tf.Variable(0, name = 'global_step',trainanle = False)
	### note use global step to track trainging step
	train_op = optimizer.minimize(loss, global_step = global_step)
	return train_op
