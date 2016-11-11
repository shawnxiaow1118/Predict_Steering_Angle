# Predict Steering angles for self driving car

# *** Use Deep Learning to Predict_Steering_Angle, Udacity challenge. ****
#
# Copyright 2016 Xiao Wang, Zhaocheng Liu
# wangxiaonku@gmail.com

This package is a tensorflow implementation for steering angle prediction
----------------------------
Commanding Running Arguments
----------------------------

Using terminal in the directory of the README file. 
Make sure that you have access to run the ./run.sh and ./run_test.sh
Surce code will be compiled and run in the bash command file

If not, type:
chmod +x ./run.sh
chmod +x ./run_test.sh

First you need to download images data from udacity and put all the image data in a folder named images within the folder ./data/, and test images into folder ./test_data/

For training purpose:
In the same directory, typing:
./run.sh learning_rate dropout batch_size weight_dacay

Four arguments:
	leraning_rate: learning rate of this network, typically with decay. Initially can set this to 0.002
	dropout      : the keep probability in each drop out layer in those fully connected network, controls overfitting, set between [0,1]
	batch_size   : the size of images in each batch for training, typically set to several hunderds
	weight_decay : another way to control the overfitting problem,  need to set carefully

For testing purpose:
In the same directory, typing:
./run_test.sh 
This should output a csv named submission.csv in current directory.

-----------
File List
------------
License 				contains license information
README 					how to run this package and some information
run.sh 					bash command to run all the training phase
run_test.sh 			bash commnad to run save model to predict for test images
/src/
    pred_steer.py       source code for the structure and essential functions for input
    read_data.py        source code read image data and all others useful data
    training.py         source code for running this deep network to train and save model.
    predict.py 			source code for running model to predict on test images
    pred_steer.pyc         
    read_data.pyc    
    predict.pyc      


/data/
     /images/      		all training images 
     anlges_train.txt 	file contains image name and its correspoding angles for trainging purpose
     anlges_valid.txt 	file contains image name and its correspoding angles for validation purpose

/test_Data/
	 /center/ 			all testing images
	 test_list.txt   	list of file name of all test images

/save/
	model saved from the training phase

/tensor/
	summary file from the training phase using tensorboard

/sampling/
	generate.py 		split the original angles into two or three parts to avoid biased data set
	train_test_split.py to split the data set before training into train set and validation set

