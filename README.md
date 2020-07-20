# MANet


## RGBT234 dataset
链接：https://pan.baidu.com/s/1weaiBh0_yH2BQni5eTxHgg 
提取码：qvsq
## RGBT210 dataset
链接：https://pan.baidu.com/s/1FClmX0SH3WarcczkEQbmwA 
提取码：ps8j 
## GTOT dataset
链接：https://pan.baidu.com/s/1zaR6aXh9PVQs063Q_b9zQg 
提取码：ajma
## RGBT234 toolkit
链接：https://pan.baidu.com/s/1UksOGtD2yl6k8mtB-Wr39A 
提取码：4f68
## RGBT210 toolkit
链接：https://pan.baidu.com/s/1KHMlbhu5R29CJvundGL4Sw 
提取码：8wtc
## GTOT toolkit
链接：https://pan.baidu.com/s/1iVVAXS4LZLvoQSGQnz7ROw 
提取码：d53m


## MANet result
MANet result in paper have upload in here, 
the report reslut is PR_0.777 SR_0.539 on RGBT234, PR_0.894 SR_0.724 on GTOT.

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

pretrain model :https://drive.google.com/open?id=1aO6LhOTxmpd7o_JXPLPjL3LsrQ5oqbl7

## Run tracker

in the tracking/run_tracker.py file  you need change dataset path  and save result file dirpath 
in the tracking/options.py file you need set model file path ,and set learning rate depend on annotation.
in tracking and train stage you need update modules/MANet3x1x1_IC.py file depend on annotation.

tracking model:https://drive.google.com/open?id=1Png508G4kQPI6HNewKQ4cfS36CvoSFSN

## Result
 ![image](https://github.com/Alexadlu/MANet/blob/master/MANet-rgbt234.png)
