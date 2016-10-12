#-*- coding: utf-8 -*-
# Read data and feeddict
#
# *** Udacity Predict steering angle ****
#
# Copyright 2016 Zhaocheng Liu, Xiao Wang
# {zcliu,  xwang696}@gatech.edu

import tensorflow as tf
import random

BATCH_SIZE = 200
HEIGHT = 480
WIDTH = 640
CHANNEL = 3
random.seed(18)
data_dir = "./image/"

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
	img = tf.image.resize_images(img,HEIGHT, WIDTH)
	tf.cast(angle, tf.float32)
	return img, angle

def input(filepath, BATCH_SIZE):
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
	input_queue = tf.train.slice_input_producer([filenames,labels])
	### read image from the input queue
	image, ang = read_data(input_queue)
	### create batch input
	images, angles = tf.train.batch([image, ang], batch_size = BATCH_SIZE)
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
	filenames = read_file_list(filepath)
	### Create file queue
	input_queue = tf.train.slice_input_producer(filenames)
	### read image from the input queue
	image,ang = read_data(input_queue)
	### flip image randomly
	distorted_image = tf.image.flip_left_right(image)
	### modify brightness and contrast with random order
	b_o_c = random.randint(0,1)
	if b_o_c == 0:
		distorted_image = tf.image.random_brightness(distorted_image, max_delta = 0.6)
		distorted_image = tf.image.random_contrast(distorted_image, lower = 0.2, upper = 0.7)
	else:
		distorted_image = tf.image.random_contrast(distorted_image, lower = 0.2, upper = 0.7)
		distorted_image = tf.image.random_brightness(distorted_image, max_delta = 0.6)

	### create batch input
	min_after_dequeue = 150
	capacity = min_after_dequeue + 3*BATCH_SIZE
	images, angles = tf.train.shuffle_batch([distorted_image, ang], batch_size = BATCH_SIZE,
		num_threads = 3, capacity = capacity, min_after_dequeue = min_after_dequeue)
	return images, angles