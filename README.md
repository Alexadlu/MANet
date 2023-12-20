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
Here, we have only uploaded the result file of the paper (PR_0.777 SR_0.539 on RGBT234, PR_0.894 SR_0.724 on GTOT.)

This code is an updated version, simplified from the one we submitted for the VOT2019-RGBT challenge.

Consequently, there are some differences compared to MANet's paper.

## Prerequisites

CPU: Intel(R) Core(TM) i7-7700K CPU @ 3.75GHz
GPU: NVIDIA GTX1080（8GB）
Ubuntu 16.04

* python2.7
* pytorch == 0.3.1
* numpy
* PIL
* some others library functions 

## Pretrained model for MANet

In our tracker, we use [MDNet](https://github.com/HyeonseobNam/py-MDNet) as our backbone and extend it to a multi-modal tracker.

We use imagenet-vgg-m.mat as our pretrain model.

## Train

You can choose either a two stage training or end2end training
### two stage train: 
* Stage1: Use the RGBT dataset to train whole network, then save the final model; 
* Stage2: Load only the parameters of GA from the model saved in Stage1, and use the same RGBT dataset to train MA and IA while keeping GA fixed.   
### end2end train:
* The training method here is the same as with [MDNet](https://github.com/HyeonseobNam/py-MDNet)

Pretrain model :https://drive.google.com/open?id=1aO6LhOTxmpd7o_JXPLPjL3LsrQ5oqbl7

## Run tracker

In the tracking/run_tracker.py file, you need to change the dataset path and save the result file directory. 
In the tracking/options.py file, you need to set model file path and set learning rate depend on annotation.
For the testing and training stages, update the 'modules/MANet3x1x1_IC.py' file depending on the annotation.

Tracking model:https://drive.google.com/open?id=1Png508G4kQPI6HNewKQ4cfS36CvoSFSN

## Result
 ![image](https://github.com/Alexadlu/MANet/blob/master/MANet-rgbt234.png)
