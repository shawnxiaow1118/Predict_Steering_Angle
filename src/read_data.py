import scipy.misc
import cv2
import scipy
import random


def read(train_file, valid_file):
	train_angles = []
	train_images = []
	valid_angles = []
	valid_images = []


	### read training data
	with open(train_file, 'r') as f:
		tmp = f.read().splitlines()
		for line in tmp:
			one_line = line.split(" ")
			train_images.append("./data/images/" + one_line[0])
			train_angles.append(float(one_line[1]))

### read validation data
	with open(valid_file, 'r') as f:
		tmp = f.read().splitlines()
		for line in tmp:
			one_line = line.split(" ")
			valid_images.append("./data/images/" + one_line[0])
			valid_angles.append(float(one_line[1]))

	return train_images, train_angles, valid_images, valid_angles
	# c_train = list(zip(train_images, train_angles))
	# c_valid = list(zip(valid_images, valid_angles))

	# random.shuffle(c_train)
	# random.shuffle(c_valid)

	# train_images, train_angles = zip(*c_train)
	# valid_images, valid_angles = zip(*c_valid)


def Train_Batch(train_images, train_angles,batch_size): #, train_pointer):
	imgs_out = []
	angs_out = []
	num_train = len(train_angles)
	for i in range(0, batch_size):
		index  = random.randint(0, num_train-1)
		imgs = scipy.misc.imread(train_images[index])[-280:]
		#imgs = scipy.misc.imread(train_images[(train_pointer+i)%num_train])[-280:]
		l_imgs = scipy.misc.imresize(imgs, [140,320])/255.0
		angle = train_angles[index]
		#angle = train_angles[(train_pointer+i)%num_train]
		imgs_out.append(l_imgs)
		angs_out.append(angle)
	#train_pointer = train_pointer + batch_size
	return imgs_out, angs_out

def Valid_Batch(valid_images, valid_angles, batch_size): #, valid_pointer):
	imgs_out = []
	angs_out = []
	num_valid = len(valid_angles)
	for i in range(0, batch_size):
		index = random.randint(0, num_valid-1)
		imgs = scipy.misc.imread(valid_images[index])[-280:]	
		#imgs = scipy.misc.imread(valid_images[(valid_pointer+i)%num_valid])[-280:]
		l_imgs = scipy.misc.imresize(imgs, [140,320])/255.0
		angle = valid_angles[index]
		#angle = valid_angles[(valid_pointer+i)%num_valid]
		imgs_out.append(l_imgs)
		angs_out.append(angle)
	#valid_pointer = valid_pointer + batch_size
	return imgs_out, angs_out

def Shuffle(images, angles):
	c = list(zip(images, angles))
	#c_valid = list(zip(valid_images, valid_angles))

	random.shuffle(c)
	#random.shuffle(c_valid)

	images, angles = zip(*c)
	#valid_images, valid_angles = zip(*c_valid)
	return images, angles


def read_image(file_name):
	imgs = scipy.misc.imread(file_name)[-280:]
	l_imgs = scipy.misc.imresize(imgs, [140,320])/255.0
	return l_imgs

def read_test(test_path):
	files = []
	f = open(test_path,'r')
	tmp = f.read().splitlines()
	for line in tmp:
		files.append(line)
	return files