# Predict Steering angles for self driving car

 Use Deep Learning to Predict_Steering_Angle, Udacity challenge

## Getting Started



### Prerequisites

tensorflow0.11.0, python2.7, numpy, pandas, scipy, cv2 if using GPU for training also CUDA and cudnn are needed.


### Running

Using terminal in the directory of the README file. Make sure that you have access to run the ./run.sh and ./run_test.sh. Source code will be compiled and run in the bash command file.

If not, type:
```
chmod +x ./run.sh
chmod +x ./run_test.sh
```

Then you need to download images data from udacity and put all the image data in a folder named images within the folder ./data/, and test images into folder ./test_data/

For training purpose, in the same directory, typing:
```
./run.sh learning_rate dropout batch_size weight_dacay
```

Four arguments:
* leraning_rate: learning rate of this network, typically with decay. Initially can set this to 0.002
* dropout      : the keep probability in each drop out layer in those fully connected network, controls overfitting, set between [0,1]
* batch_size   : the size of images in each batch for training, typically set to several hunderds
* weight_decay : another way to control the overfitting problem,  need to set carefully

For testing purpose, in the same directory, typing:
```
./run_test.sh 
```
This should output a csv named submission.csv in current directory.

## Built With

* [Tensorflow](https://www.tensorflow.org) - The computation framework and platform used
 
## Authors

* **Xiao Wang** 
* **Zhaocheng Liu** 


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* [Nvidia paper](https://arxiv.org/pdf/1604.07316.pdf)- Original Paper



