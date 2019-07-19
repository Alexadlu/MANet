import numpy as np
import os
import sys
import time
import argparse
import json
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable

sys.path.insert(0,'../modules')
from sample_generator import *
from data_prov import *
from MANet3x1x1_IC import *
from bbreg import *
from options import *
from gen_config import *


def forward_samples(model, image1,image2, samples, out_layer='conv3'):
    model.eval()
    extractor1 = RegionExtractor(image1, samples, opts['img_size'], opts['padding'], opts['batch_test'])
    extractor2 = RegionExtractor(image2, samples, opts['img_size'], opts['padding'], opts['batch_test'])
    for i, regions1 in enumerate(extractor1):
        for j, regions2 in enumerate(extractor2):
            if i==j:
                regions1 = Variable(regions1)
                regions2 = Variable(regions2)
                if opts['use_gpu']:
                    regions1 = regions1.cuda()
                    regions2 = regions2.cuda()
                feat = model(regions1, regions2,out_layer=out_layer)
                if i==0:
                    feats = feat.data.clone()
                else:
                    feats = torch.cat((feats,feat.data.clone()),0)
    return feats



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


def train(model, criterion, optimizer, pos_feats, neg_feats, maxiter, in_layer='fc4'):
    model.train()
    
    batch_pos = opts['batch_pos']
    batch_neg = opts['batch_neg']
    batch_test = opts['batch_test']
    batch_neg_cand = max(opts['batch_neg_cand'], batch_neg)

    pos_idx = np.random.permutation(pos_feats.size(0))
    neg_idx = np.random.permutation(neg_feats.size(0))
    while(len(pos_idx) < batch_pos*maxiter):
        pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_feats.size(0))])
    while(len(neg_idx) < batch_neg_cand*maxiter):
        neg_idx = np.concatenate([neg_idx, np.random.permutation(neg_feats.size(0))])
    pos_pointer = 0
    neg_pointer = 0

    for iter in range(maxiter):

        # select pos idx
        pos_next = pos_pointer+batch_pos
        pos_cur_idx = pos_idx[pos_pointer:pos_next]
        pos_cur_idx = pos_feats.new(pos_cur_idx).long()
        pos_pointer = pos_next

        # select neg idx
        neg_next = neg_pointer+batch_neg_cand
        neg_cur_idx = neg_idx[neg_pointer:neg_next]
        neg_cur_idx = neg_feats.new(neg_cur_idx).long()
        neg_pointer = neg_next

        # create batch
        batch_pos_feats = Variable(pos_feats.index_select(0, pos_cur_idx))
        batch_neg_feats = Variable(neg_feats.index_select(0, neg_cur_idx))

        # hard negative mining
        if batch_neg_cand > batch_neg:
            model.eval()
            for start in range(0,batch_neg_cand,batch_test):
                end = min(start+batch_test,batch_neg_cand)
                score = model(feat=batch_neg_feats[start:end],in_layer=in_layer)
                if start==0:
                    neg_cand_score = score.data[:,1].clone()
                else:
                    neg_cand_score = torch.cat((neg_cand_score, score.data[:,1].clone()),0)

            _, top_idx = neg_cand_score.topk(batch_neg)
            batch_neg_feats = batch_neg_feats.index_select(0, Variable(top_idx))
            model.train()
        
        # forward
        pos_score = model(feat=batch_pos_feats, in_layer=in_layer)
        neg_score = model(feat=batch_neg_feats, in_layer=in_layer)
        
        # optimize
        loss = criterion(pos_score, neg_score)
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), opts['grad_clip'])
        optimizer.step()

def run_mdnet(img_list1,img_list2, init_bbox, gt=None, savefig_dir='', display=False):

    # Init bbox
    target_bbox = np.array(init_bbox)
    result = np.zeros((len(img_list1),4))
    result_bb = np.zeros((len(img_list1),4))
    result[0] = target_bbox
    result_bb[0] = target_bbox

    # Init model
    model = MDNet(opts['model_path1'])
    if opts['use_gpu']:
        model = model.cuda()
    model.set_learnable_params(opts['ft_layers'])
    
    # Init criterion and optimizer 
    criterion = BinaryLoss()
    init_optimizer = set_optimizer(model, opts['lr_init'])
    update_optimizer = set_optimizer(model, opts['lr_update'])

    tic = time.time()
    # Load first image
    image1 = Image.open(img_list1[0]).convert('RGB')
    image2 = Image.open(img_list2[0]).convert('RGB')

    # Train bbox regressor
    bbreg_examples = gen_samples(SampleGenerator('uniform', image1.size, 0.3, 1.5, 1.1),
                                 target_bbox, opts['n_bbreg'], opts['overlap_bbreg'], opts['scale_bbreg'])
    bbreg_feats = forward_samples(model, image1, image2,bbreg_examples)
    bbreg = BBRegressor(image1.size)
    bbreg.train(bbreg_feats, bbreg_examples, target_bbox)

    # Draw pos/neg samples
    pos_examples = gen_samples(SampleGenerator('gaussian', image1.size, 0.1, 1.2),
                               target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])

    neg_examples = np.concatenate([
                    gen_samples(SampleGenerator('uniform', image1.size, 1, 2, 1.1), 
                                target_bbox, opts['n_neg_init']//2, opts['overlap_neg_init']),
                    gen_samples(SampleGenerator('whole', image1.size, 0, 1.2, 1.1),
                                target_bbox, opts['n_neg_init']//2, opts['overlap_neg_init'])])
    neg_examples = np.random.permutation(neg_examples)

    # Extract pos/neg features
    pos_feats = forward_samples(model, image1, image2,pos_examples)
    neg_feats = forward_samples(model, image1, image2,neg_examples)
    feat_dim = pos_feats.size(-1)

    # Initial training
    train(model, criterion, init_optimizer, pos_feats, neg_feats,opts['maxiter_init'])
    
    # Init sample generators
    sample_generator = SampleGenerator('gaussian', image1.size, opts['trans_f'], opts['scale_f'], valid=True)
    pos_generator = SampleGenerator('gaussian', image1.size, 0.1, 1.2)
    neg_generator = SampleGenerator('uniform', image1.size, 1.5, 1.2)

    # Init pos/neg features for update
    pos_feats_all = [pos_feats[:opts['n_pos_update']]]
    neg_feats_all = [neg_feats[:opts['n_neg_update']]]
  
    
    spf_total = time.time()-tic

    # Display
    savefig = savefig_dir != ''
    if display or savefig: 
        dpi = 80.0
        figsize = (image1.size[0]/dpi, image1.size[1]/dpi)

        fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        im = ax.imshow(image1, aspect='normal')

        if gt is not None:
            gt_rect = plt.Rectangle(tuple(gt[0,:2]),gt[0,2],gt[0,3], 
                    linewidth=3, edgecolor="#00ff00", zorder=1, fill=False)
            ax.add_patch(gt_rect)
        
        rect = plt.Rectangle(tuple(result_bb[0,:2]),result_bb[0,2],result_bb[0,3], 
                linewidth=3, edgecolor="#ff0000", zorder=1, fill=False)
        ax.add_patch(rect)

        if display:
            plt.pause(.01)
            plt.draw()
        if savefig:
            fig.savefig(os.path.join(savefig_dir,'0000.jpg'),dpi=dpi)
    
    # Main loop
    count=0
    for i in range(1,len(img_list1)):
        
        tic = time.time()
        # Load image
        image1 = Image.open(img_list1[i]).convert('RGB')# RGB
        image2 = Image.open(img_list2[i]).convert('RGB')# T
        # Estimate target bbox
        samples = gen_samples(sample_generator, target_bbox, opts['n_samples'])
        sample_scores = forward_samples(model, image1, image2,samples, out_layer='fc6')
        top_scores, top_idx = sample_scores[:,1].topk(5)
        top_idx = top_idx.cpu().numpy()
        target_score = top_scores.mean()
        target_bbox = samples[top_idx].mean(axis=0)

        success = target_score > opts['success_thr']
        
        # Expand search area at failure
        if success:
            count=0
            sample_generator.set_trans_f(opts['trans_f'])
        else:
            count=count+1
            sample_generator.set_trans_f(opts['trans_f_expand'])

        # Bbox regression
        if success:
            bbreg_samples = samples[top_idx]
            bbreg_feats = forward_samples(model, image1, image2,bbreg_samples)

            bbreg_samples = bbreg.predict(bbreg_feats, bbreg_samples)
            bbreg_bbox = bbreg_samples.mean(axis=0)
        else:
            bbreg_bbox = target_bbox
        
        # Copy previous result at failure
        if not success:

            target_bbox = result[i-1]
            bbreg_bbox = result_bb[i-1]

                
        # Save result
        result[i] = target_bbox
        result_bb[i] = bbreg_bbox

        # Data collect
        if success:
            # Draw pos/neg samples
            pos_examples = gen_samples(pos_generator, target_bbox, 
                                       opts['n_pos_update'],
                                       opts['overlap_pos_update'])
            neg_examples = gen_samples(neg_generator, target_bbox, 
                                       opts['n_neg_update'],
                                       opts['overlap_neg_update'])

            # Extract pos/neg features
            pos_feats = forward_samples(model, image1, image2,pos_examples)
            neg_feats = forward_samples(model, image1, image2,neg_examples)
            pos_feats_all.append(pos_feats)
            neg_feats_all.append(neg_feats)
            if len(pos_feats_all) > opts['n_frames_long']:
                del pos_feats_all[0]
            if len(neg_feats_all) > opts['n_frames_short']:
                del neg_feats_all[0]

        # Short term update
        if not success:
            nframes = min(opts['n_frames_short'],len(pos_feats_all))
            pos_data = torch.stack(pos_feats_all[-nframes:],0).view(-1,feat_dim)
            neg_data = torch.stack(neg_feats_all,0).view(-1,feat_dim)
            train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])
        
        # Long term update
        elif i % opts['long_interval'] == 0:
            pos_data = torch.stack(pos_feats_all,0).view(-1,feat_dim)
            neg_data = torch.stack(neg_feats_all,0).view(-1,feat_dim)
            train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])
        
        spf = time.time()-tic
        spf_total += spf

        # Display
        if display or savefig:
            im.set_data(image1)

            if gt is not None:
                gt_rect.set_xy(gt[i,:2])
                gt_rect.set_width(gt[i,2])
                gt_rect.set_height(gt[i,3])

            rect.set_xy(result_bb[i,:2])
            rect.set_width(result_bb[i,2])
            rect.set_height(result_bb[i,3])
            
            if display:
                plt.pause(.01)
                plt.draw()
            if savefig:
                fig.savefig(os.path.join(savefig_dir,'%04d.jpg'%(i)),dpi=dpi)

        if gt is None:
            print "Frame %d/%d, Score %.3f, Time %.3f" % \
                (i, len(img_list1), target_score, spf)
        else:
            print "Frame %d/%d, Overlap %.3f, Score %.3f, Time %.3f" % \
                (i, len(img_list1), overlap_ratio(gt[i],result_bb[i])[0], target_score, spf)

    fps = len(img_list1) / spf_total
    return result, result_bb, fps


def get_sequence(seq,seq_home):
    result_home='../MANet-RGBT234_result' # you need creat result dirpath
    img_dir1=os.path.join(seq_home,seq,'visible')
    img_dir2=os.path.join(seq_home,seq,'infrared')
    gt_path=os.path.join(seq_home,seq,'init.txt')
    img_list1=os.listdir(img_dir1)
    img_list1.sort()
    img_list1=[os.path.join(img_dir1,x) for x in img_list1]
    img_list2=os.listdir(img_dir2)
    img_list2.sort()
    img_list2=[os.path.join(img_dir2,x) for x in img_list2]
    with open(gt_path) as f:
        gt=np.loadtxt((x.replace(',',' ')for x in f))
    init_bbox=gt[0]

    result_dir=os.path.join(result_home)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_path=os.path.join(result_dir,'MANet311-2IC_'+seq+'.txt')
    return img_list1,img_list2,init_bbox,gt,result_path

data_dir="/home/adlu/adlu_work1/RGBT-MdNet/dataset/RGB-T234/" # set tracking dataset path

res_dir='../MANet-RGBT234_result' # you need creat result dirpath



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--display', action='store_true')
    parser.add_argument('-s', '--seq', default='', help='input seq')
    args = parser.parse_args()
    seq = args.seq
    
    tracker_name='result'

    list_path=os.path.join('../rgbt234.txt')

    seqs=[]

    total_fps=0
    with open(list_path) as f:
        content=f.readlines()
    for line in content:
        parsed_line=line.split()
        seqs.append(parsed_line[0])
    
    n_seq=len(seqs)
    for i in range(n_seq):
        
        seq=seqs[i]
        print(i,seq)
        if 'MANet311-2IC_'+seq+'.txt' not in os.listdir(res_dir):
            np.random.seed(123)
            torch.manual_seed(456)
            torch.cuda.manual_seed(789)
        
            img_list1,img_list2,init_bbox,gt,result_path=get_sequence(seq,data_dir)

            # Run tracker
            result, result_bb, fps = run_mdnet(img_list1, img_list2,init_bbox, gt=gt, savefig_dir='', display=args.display)
            print('FPS==>',fps)

            thresholds=np.arange(0,1.05,0.05)
            n_frame=len(gt)

            success=np.zeros(n_frame)
            iou=np.zeros(n_frame)
            for i in range(n_frame):
                iou[i]=overlap_ratio(gt[i],result_bb[i])
            for i in range(len(thresholds)):
                success[i]=sum(iou>thresholds[i])/n_frame


        # Save result
            f=open(result_path,'w+')
            for i in range(len(result_bb)):
                res='{} {} {} {} {} {} {} {}'.format(result_bb[i][0],result_bb[i][1],
                
                                                    result_bb[i][0]+result_bb[i][2],result_bb[i][1],
                
                                                    result_bb[i][0]+result_bb[i][2],result_bb[i][1]+result_bb[i][3],
                                                    
                                                    result_bb[i][0],result_bb[i][1]+result_bb[i][3]
                ) 
                f.write(res)
                f.write('\n')
            f.close()
        
