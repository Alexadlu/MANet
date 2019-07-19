import os
import sys
import pickle
import time

import torch
import torch.optim as optim
from torch.autograd import Variable

from data_prov import *
from MANet3x1x1_IC import *
from options import *

#********************************************set dataset path ********************************************
#********************************************set seq list .pkl file path  ********************************
img_home = "/home/adlu/adlu_work1/RGBT-MdNet/dataset/RGB-T234/"
data_path1 = 'data/rgbt234_v.pkl'
data_path2 = 'data/rgbt234_I.pkl'
#*********************************************************************************************************


def set_optimizer(model, lr_base, lr_mult=opts['lr_mult'], momentum=opts['momentum'], w_decay=opts['w_decay']):
    params = model.get_learnable_params()
    param_list = []
    for k, p in params.iteritems():
        lr = lr_base
        for l, m in lr_mult.iteritems():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr':lr})
    optimizer = optim.SGD(param_list, lr = lr, momentum=momentum, weight_decay=w_decay)
    return optimizer


def train_mdnet():
    
    ## Init dataset ##
    with open(data_path1, 'rb') as fp1:
        data1 = pickle.load(fp1)

    K1 = len(data1)
    dataset1 = [None]*K1
    for k, (seqname, seq) in enumerate(sorted(data1.iteritems())):
        img_list1 = seq['images']
        gt1 = seq['gt']
        img_dir1 = os.path.join(img_home, seqname)
        dataset1[k] = RegionDataset(img_dir1, img_list1, gt1, opts)


    with open(data_path2,'rb') as fp2:
        data2=pickle.load(fp2)

    K2=len(data2)
    dataset2=[None]*K2
    for k ,(seqname,seq) in enumerate(sorted(data2.iteritems())):
        pos_regions,neg_regions,pos_examples,neg_examples,idx=dataset1[k].next()

        img_list2 = seq['images']
        gt2 = seq['gt']
        img_dir2 = os.path.join(img_home, seqname)
        dataset2[k] = RegionDataset1(img_dir2, img_list2, gt2,pos_regions,neg_regions,pos_examples,neg_examples,idx ,opts)



    ## Init model ##
    model = MDNet(opts['init_model_path'], K1)
    if opts['use_gpu']:
        model = model.cuda()
    model.set_learnable_params(opts['ft_layers'])
        
    ## Init criterion and optimizer ##
    criterion = BinaryLoss()
    evaluator = Precision()
    optimizer = set_optimizer(model, opts['lr'])

    best_prec = 0.
    for i in range(opts['n_cycles']):
        print "==== Start Cycle %d ====" % (i)
        k_list = np.random.permutation(K1)
        prec = np.zeros(K1)
        for j,k in enumerate(k_list):
            tic = time.time()
            pos_regions1, neg_regions1,pos_examples1,neg_examples1,idx1,pos_regions2, neg_regions2,pos_examples2,neg_examples2,idx2 = dataset2[k].next1()
            
            pos_regions1 = Variable(pos_regions1)
            neg_regions1 = Variable(neg_regions1)
            pos_regions2 = Variable(pos_regions2)
            neg_regions2 = Variable(neg_regions2)

        
            if opts['use_gpu']:
                pos_regions1 = pos_regions1.cuda()
                neg_regions1 = neg_regions1.cuda()
                pos_regions2 = pos_regions2.cuda()
                neg_regions2 = neg_regions2.cuda()
        
            pos_score = model(pos_regions1,pos_regions2 ,k)
            neg_score = model(neg_regions1,neg_regions2, k)

            loss = criterion(pos_score, neg_score)
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), opts['grad_clip'])
            optimizer.step()
            
            prec[k] = evaluator(pos_score, neg_score)

            toc = time.time()-tic
            print "Cycle %2d, K %2d (%2d), Loss %.3f, Prec %.3f, Time %.3f" % \
                    (i, j, k, loss.data[0], prec[k], toc)

        cur_prec = prec.mean()
        print "Mean Precision: %.3f" % (cur_prec)
        if cur_prec > best_prec:
            best_prec = cur_prec
            if opts['use_gpu']:
                model = model.cpu()
            states = {
                    'shared_layers': model.layers.state_dict(),

                    'RGB_para1_3x3': model.RGB_para1_3x3.state_dict(),
                    'RGB_para2_1x1': model.RGB_para2_1x1.state_dict(),
                    'RGB_para3_1x1': model.RGB_para3_1x1.state_dict(),

                    'T_para1_3x3': model.T_para1_3x3.state_dict(),
                    'T_para2_1x1': model.T_para2_1x1.state_dict(),
                    'T_para3_1x1': model.T_para3_1x1.state_dict(),
                    }


            print "Save model to %s" % opts['model_path']
            torch.save(states, opts['model_path'])
            if opts['use_gpu']:
                model = model.cuda()


if __name__ == "__main__":
    train_mdnet()