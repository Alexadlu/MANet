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
Here, we only upload the result file of paper (PR_0.777 SR_0.539 on RGBT234, PR_0.894 SR_0.724 on GTOT.)

This code is update version based on submitted for VOT2019-RGBT challenge code simplified version.

So there are some differences from MANet's paper.

## Prerequisites

CPU: Intel(R) Core(TM) i7-7700K CPU @ 3.75GHz
GPU: NVIDIA GTX1080
Ubuntu 16.04

* python2.7
* pytorch == 0.3.1
* numpy
* PIL
* some others library functions 

## Pretrained model for MANet

In our tracker, we use [MDNet](https://github.com/HyeonseobNam/py-MDNet) as our backbone and extend to multi-modal tracker.

We use imagenet-vgg-m.mat as our pretrain model.

## Train

You can choose two stage train or end2end train
### two stage train: 
* Stage1. use RGBT dataset to train all network, and then save finally model; 
* Stage2. you only need to load the parameters of GA from the stage1 saved model, and use same RGBT dataset to train the MA and IA while fix GA.   
### end2end train:
* Here train method is same with [MDNet](https://github.com/HyeonseobNam/py-MDNet)

Pretrain model :https://drive.google.com/open?id=1aO6LhOTxmpd7o_JXPLPjL3LsrQ5oqbl7

## Run tracker

In the tracking/run_tracker.py file, you need to change dataset path and save result file dirpath 
In the tracking/options.py file, you need to set model file path and set learning rate depend on annotation.
In tracking and train stages, you need to update modules/MANet3x1x1_IC.py file depend on annotation.

Tracking model:https://drive.google.com/open?id=1Png508G4kQPI6HNewKQ4cfS36CvoSFSN

## Result
 ![image](https://github.com/Alexadlu/MANet/blob/master/MANet-rgbt234.png)
