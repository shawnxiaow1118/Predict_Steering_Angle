#-*- coding: utf-8 -*-
# Model for predict steering
#
# *** Udacity Predict steering angle ****
#
# Copyright 2016 Zhaocheng Liu, Xiao Wang
# {zcliu,  xwang696}@gatech.edu

import tensorflow as tf

def _variable_withweight_decay(name, shape, stddev, wd):
	""" Helper function to create an initialized Varibale with weight decay
	Iuput:
		name: the name of the variable
		shape: the shape of the variable, list of ints
		stddev: stadard deviation of a truncated Gaussian
		we: add L2Loss weight decay numtiplied by this float. If None, weight 
		decay is 

	"""


def inference(images):
	""" 
	Input: batch images,
	Output: steering angle
	"""
	with tf.variable_scope('conv1') as scope:
		weights = tf.Variable(
			tf.truncated_normal([5,5,3,24]), stddev = 5e-2, name='weights')
		biases = tf.Variable(tf.zeros([24]),
                         name='biases')

		conv = tf.nn.conv2d(images, weigths,[1,2,2,1],padding = 'VALID')
		bias = tf.nn.bias_add(conv, biases)
		conv_non = tf.nn.relu(bias, name = scope.name)


	with tf.variable_scope('conv2') as scope:
		weights = tf.Variable(
			tf.truncated_normal([5,5,24,36]), stddev = 5e-2, name='weights')
		biases = tf.Variable(tf.zeros([36]),
                         name='biases')

		conv = tf.nn.conv2d(conv_non, weigths,[1,2,2,1],padding = 'VALID')
		bias = tf.nn.bias_add(conv, biases)
		conv_non = tf.nn.relu(bias, name = scope.name)

	with tf.variable_scope('conv3') as scope:
		weights = tf.Variable(
			tf.truncated_normal([5,5,36,48]), stddev = 5e-2, name='weights')
		biases = tf.Variable(tf.zeros([48]),
                         name='biases')

		conv = tf.nn.conv2d(conv_non, weigths,[1,2,2,1],padding = 'VALID')
		bias = tf.nn.bias_add(conv, biases)
		conv_non = tf.nn.relu(bias, name = scope.name)

	with tf.variable_scope('conv4') as scope:
		weights = tf.Variable(
			tf.truncated_normal([3,3,48,64]), stddev = 5e-2, name='weights')
		biases = tf.Variable(tf.zeros([64]),
                         name='biases')

		conv = tf.nn.conv2d(conv_non, weigths,[1,2,2,1],padding = 'VALID')
		bias = tf.nn.bias_add(conv, biases)
		conv_non = tf.nn.relu(bias, name = scope.name)

	with tf.variable_scope('conv5') as scope:
		weights = tf.Variable(
			tf.truncated_normal([3,3,48,64]), stddev = 5e-2, name='weights')
		biases = tf.Variable(tf.zeros([64]),
                         name='biases')

		conv = tf.nn.conv2d(conv_non, weigths,[1,2,2,1],padding = 'VALID')
		bias = tf.nn.bias_add(conv, biases)
		conv_non = tf.nn.relu(bias, name = scope.name)

	with tf.variable_scope('local6') as scope:
		reshape = tf.reshape(conv_non,[bathc_size,-1])
		dim = reshape.get_shape()[1].value
		weights = tf.Variable(
			tf.truncated_normal([dim, 100]), stddev = 5e-2, name='weights')
		biases = tf.Variable(tf.zeros([100]),
                         name='biases')
		local6 = tf.nn.relu(tf.matmul(reshape, weights)+biases, name = scope.name)



	with tf.variable_scope('local7') as scope:
		weights = tf.Variable(
			tf.truncated_normal([100, 50]), stddev = 5e-2, name='weights')
		biases = tf.Variable(tf.zeros([50]),
                         name='biases')
		local7 = tf.nn.relu(tf.matmul(reshape, weights)+biases, name = scope.name)

	with tf.variable_scope('local8') as scope:
		weights = tf.Variable(
			tf.truncated_normal([50, 10]), stddev = 5e-2, name='weights')
		biases = tf.Variable(tf.zeros([10]),
                         name='biases')
		local8 = tf.nn.relu(tf.matmul(reshape, weights)+biases, name = scope.name)

	with tf.variable_scope('linear_regression') as scope:
		weights = tf.Variable(
			tf.truncated_normal([10, 1]), stddev = 5e-2, name='weights')
		biases = tf.Variable(tf.zeros([1]),
                         name='biases')
		output = tf.matmul(reshape, weights)+biases
	return output
		