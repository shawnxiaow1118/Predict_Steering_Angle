#-*- coding: utf-8 -*-
# Model for predict steering
#
# *** Udacity Predict steering angle ****
#
# Copyright 2016 Xiao Wang, Zhaocheng Liu
# {xwang696, zcliu}@gatech.edu

import tensorflow as tf


def _variable_withweight_decay(name, shape, stddev, wd):
	""" Helper function to create an initialized Varibale with weight decay
	Iuput:
		name: the name of the variable
		shape: the shape of the variable, list of ints
		stddev: stadard deviation of a truncated Gaussian
		we: add L2 Loss weight decay numtiplied by this float. If None, weight 
		decay is 

	"""
batch_size = 10

def inference(images):
	""" 
	Input: batch images,
	Output: steering angle
	"""

	with tf.variable_scope('conv1') as scope:
		weights = tf.Variable(
			tf.truncated_normal([5,5,3,24], stddev = 5e-2, name='weights'))
		biases = tf.Variable(tf.zeros([24]),
                         name='biases')

		conv = tf.nn.conv2d(images, weights,[1,2,2,1],padding = 'VALID')
		bias = tf.nn.bias_add(conv, biases)
		conv_non = tf.nn.relu(bias, name = scope.name)


	with tf.variable_scope('conv2') as scope:
		weights = tf.Variable(
			tf.truncated_normal([5,5,24,36], stddev = 5e-2, name='weights'))
		biases = tf.Variable(tf.zeros([36]),
                         name='biases')

		conv = tf.nn.conv2d(conv_non, weights,[1,2,2,1],padding = 'VALID')
		bias = tf.nn.bias_add(conv, biases)
		conv_non = tf.nn.relu(bias, name = scope.name)

	with tf.variable_scope('conv3') as scope:
		weights = tf.Variable(
			tf.truncated_normal([5,5,36,48], stddev = 5e-2, name='weights'))
		biases = tf.Variable(tf.zeros([48]),
                         name='biases')

		conv = tf.nn.conv2d(conv_non, weights,[1,2,2,1],padding = 'VALID')
		bias = tf.nn.bias_add(conv, biases)
		conv_non = tf.nn.relu(bias, name = scope.name)

	with tf.variable_scope('conv4') as scope:
		weights = tf.Variable(
			tf.truncated_normal([5,5,48,64], stddev = 5e-2, name='weights'))
		biases = tf.Variable(tf.zeros([64]),
                         name='biases')

		conv = tf.nn.conv2d(conv_non, weights,[1,2,2,1],padding = 'VALID')
		bias = tf.nn.bias_add(conv, biases)
		conv_non = tf.nn.relu(bias, name = scope.name)

	with tf.variable_scope('conv5') as scope:
		weights = tf.Variable(
			tf.truncated_normal([5,5,64,64], stddev = 5e-2, name='weights'))
		biases = tf.Variable(tf.zeros([64]),
                         name='biases')

		conv = tf.nn.conv2d(conv_non, weights,[1,2,2,1],padding = 'VALID')
		bias = tf.nn.bias_add(conv, biases)
		conv_non = tf.nn.relu(bias, name = scope.name)

	with tf.variable_scope('conv6') as scope:
		weights = tf.Variable(
			tf.truncated_normal([5,5,64,64], stddev = 5e-2, name='weights'))
		biases = tf.Variable(tf.zeros([64]),
                         name='biases')

		conv = tf.nn.conv2d(conv_non, weights,[1,2,2,1],padding = 'VALID')
		bias = tf.nn.bias_add(conv, biases)
		conv_non = tf.nn.relu(bias, name = scope.name)

	with tf.variable_scope('local7') as scope:
		reshape = tf.reshape(conv_non,[batch_size,-1])
		dim = reshape.get_shape()[1].value
		weights = tf.Variable(
			tf.truncated_normal([dim, 100], stddev = 5e-2, name='weights'))
		biases = tf.Variable(tf.zeros([100]),
                         name='biases')
		local7 = tf.nn.relu(tf.matmul(reshape, weights)+biases, name = scope.name)



	with tf.variable_scope('local8') as scope:
		weights = tf.Variable(
			tf.truncated_normal([100, 50], stddev = 5e-2, name='weights'))
		biases = tf.Variable(tf.zeros([50]),
                         name='biases')
		local8 = tf.nn.relu(tf.matmul(local7, weights)+biases, name = scope.name)

	with tf.variable_scope('local9') as scope:
		weights = tf.Variable(
			tf.truncated_normal([50, 10], stddev = 5e-2, name='weights'))
		biases = tf.Variable(tf.zeros([10]),
                         name='biases')
		local9 = tf.nn.relu(tf.matmul(local8, weights)+biases, name = scope.name)

	with tf.variable_scope('linear_regression') as scope:
		weights = tf.Variable(
			tf.truncated_normal([10, 1], stddev = 5e-2, name='weights'))
		biases = tf.Variable(tf.zeros([1]),
                         name='biases')
		output = tf.matmul(local9, weights)+biases
	return output
		

def loss(output, angles):
	""" Calculate the RMSE of prediction angles and labeled angles
	Args:
		output:  prediction tensor, float [batch_size]
		angles:  label tesor, float [batch_size]

	Returns:
		loss: Loss tensor of type float
	"""
	reshape = tf.reshape(angles, [batch_size,1])
	angs = tf.string_to_number(angles, out_type = tf.float32)
	### calculate square sum
	sum_square = tf.square(tf.sub(output, angs))
	### calculate mean square mean
	mean_square = tf.reduce_mean(sum_square)
	### take root
	loss = tf.sqrt(mean_square)
	return loss

def train(loss, learning_rate, global_step):
	""" Setting the training operation
	Args: 
		loss: loss tensor, from loss()
		learning_rate: the learning reate for gradient descent 
	"""
	### creat the optimizer using learing_rate
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	###  creat a global variable
	# global_step = tf.Variable(0, name = 'global_step',trainanle = False)
	### note use global step to track trainging step
	train_op = optimizer.minimize(loss)
	return train_op
