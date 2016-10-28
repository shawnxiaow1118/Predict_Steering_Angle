#-*- coding: utf-8 -*-
# Read data and feeddict
#
# *** Udacity Predict steering angle ****
#
# Copyright 2016 Zhaocheng Liu, Xiao Wang
# {zcliu,  xwang696}@gatech.edu

import tensorflow as tf
import random

HEIGHT = 480
WIDTH = 640
CHANNEL = 3
#random.seed(18)
data_dir = "./center_camera/"

def read_file_list(list_file):
	""" Read image file names and angles from list file
	Args:
		list_file: a txt file contains list of image files' name

	Returns:
		a list of filenames and labels
	"""
	f = open(list_file,'r')
	temp = f.read().splitlines()
	filenames = []
	labels = []
	for line in temp:
		one_line = line.split(" ")
		filenames.append(data_dir+one_line[0])
		labels.append(one_line[1])
	return filenames, labels

def read_data(input_queue):
	""" Read and decode a single image from input queue
	Args:
		filename_queue: a queue contains file name, A scalar string tensor

	Returns:
		two tensors: the decoded image and float angles
	"""
	#reader = tf.WholeFileReader()
	angle = input_queue[1]
	#key, value = reader.read(input_queue[0])
	value = tf.read_file(input_queue[0])
	img = tf.image.decode_jpeg(value, channels = CHANNEL)
	img = tf.image.resize_images(img,[HEIGHT, WIDTH])
	img = img[:,:,:]/255.0
	tf.cast(angle, tf.float32)
	return img, angle

def origin_inputs(filepath, BATCH_SIZE):
	""" Provide batch images and angles
	Args: 
		filepath: contains the path to the file containing list of images
		BATCH_SIZE: the size of the batch provided
		Epochs: a maximum number of epochs.

	Returns:
		 A input data tensor with first dimension as batch_size and a tensor for the angls
	"""
	### Read filelist 
	filenames, labels = read_file_list(filepath)
	### Create file queue
	input_queue = tf.train.slice_input_producer([filenames,labels],shuffle=True)
	### read image from the input queue
	image, ang = read_data(input_queue)
	### transfer labels from string to float
	n_ang = tf.string_to_number(ang, out_type=tf.float32)
	### create batch input
	images, angles = tf.train.batch([image, n_ang], batch_size = BATCH_SIZE, name = 'eval_input')
	return images, angles

#filenames,labels = read_file_list("./image/angles.txt")
# init_op = tf.initialize_all_variables()
# filename_queue = tf.train.string_input_producer(filenames)
# image = read_data(filename_queue)
# images = tf.train.batch([image],batch_size = 12)

# with tf.Session() as sess:
# 	sess.run(init_op)
# 	coord = tf.train.Coordinator()
# 	threads = tf.train.start_queue_runners(coord=coord)
# 	for i in range(1):
# 		im = sess.run(images)
# 		print(im.shape)

# 	coord.request_stop()
#	coord.join(threads)

def distorted_inputs(filepath, BATCH_SIZE):
	""" Provide batch distorted images and angles
	Args: 
		filepath: contains the path to the file containing list of images
		BATCH_SIZE: the size of the batch provided

	Returns:
		 A input data tensor with first dimension as batch_size and a tensor for the angls
	"""
	### Read filelist 
	filenames, labels = read_file_list(filepath)
	### Create file queue
	input_queue = tf.train.slice_input_producer([filenames,labels],shuffle=True)
	### read image from the input queue
	image, ang = read_data(input_queue)

	n_ang = tf.string_to_number(ang, out_type=tf.float32)
	rand = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32)
	x = tf.constant(0.6)

	distorted_image = tf.cond(tf.less(rand, x), lambda: tf.image.flip_left_right(image), lambda: image)
	n_ang = tf.cond(tf.less(rand, x), lambda: -n_ang, lambda: n_ang)

	### modify brightness and contrast with random order
	# b_o_c = random.randint(0,1)
	# if b_o_c == 0:
	# 	distorted_image = tf.image.random_brightness(distorted_image, max_delta = 0.6)
	# 	distorted_image = tf.image.random_contrast(distorted_image, lower = 0.2, upper = 0.7)
	# else:
	# 	distorted_image = tf.image.random_contrast(distorted_image, lower = 0.2, upper = 0.7)
	# 	distorted_image = tf.image.random_brightness(distorted_image, max_delta = 0.6)

	### create batch input
	# min_after_dequeue = 150
	# capacity = min_after_dequeue + 3*BATCH_SIZE

	images, angles = tf.train.batch([distorted_image, n_ang], batch_size = BATCH_SIZE,num_threads = 3, name = "distorted_input")
	return images, angles