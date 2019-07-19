# MANet
Multi-Adapter RGBT Tracking  implementation on Pytorch

this code is update version based on submitted for VOT RGBT race code simplified version.
So there are some differences from MANET's paper. 

## Prerequisites

CPU: Intel(R) Core(TM) i7-7700K CPU @ 3.75GHz
GPU: NVIDIA GTX1080
Ubuntu 16.04

* python2.7
* pytorch == 0.3.1
* numpy
* PIL
* by yourself need install some library functions 

## Pretrained model for MANet

In our tracker, we use an VGG-M Net variant as our backbone, which is end-to-end trained for visual tracking.

The train on gtot model file in models folder,name called MANet-2IC.pth ,you can use this tracking rgbt234

Then,You need to modify the path in the tracking/options.py file depending on where the file is placed. 
It is best to use an absolute path.
you can change code version of CPU/GPU in this flie

## Train

you can use RGBT dataset as train data , in pretrain floder you need 
first genrate sequence list .pkl file use prepro_data.py ,
sencod change your data path ,
fainlly excute train.py

## Run tracker

in the tracking/run_tracker.py file  you need change dataset path  and save result file dirpath 
in the tracking/options.py file you need set model file path ,and set learning rate depend on annotation.
in tracking and train stage you need update modules/MANet3x1x1_IC.py file depend on annotation.

## Result
![image] (https://github.com/Alexadlu/MANet/blob/master/MANet-rgbt234.png)
