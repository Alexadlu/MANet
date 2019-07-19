import os
import numpy as np
import pickle
from collections import OrderedDict




seq_home='../dataset/RGB-T234/'#
seqlist_path='data/rgbt234.txt'#
output_path='data/rgbt234_I.pkl'# 

visible='/visible'
infrared='/infrared'

gt_path='/init.txt'

with open(seqlist_path,'r') as fp:
    seq_list = fp.read().splitlines()

data = {}
for i,seq in enumerate(seq_list):
    seq=seq+infrared
    #seq=seq+visible
    img_list = sorted([p for p in os.listdir(seq_home+seq) if os.path.splitext(p)[1] == '.jpg'])
    (filepath,tempfilename)=os.path.split(seq)
    gt = np.loadtxt(seq_home+filepath+gt_path,delimiter=',')
    assert len(img_list) == len(gt), "Lengths do not match!!"
    
    if gt.shape[1]==8:
        x_min = np.min(gt[:,[0,2,4,6]],axis=1)[:,None]
        y_min = np.min(gt[:,[1,3,5,7]],axis=1)[:,None]
        x_max = np.max(gt[:,[0,2,4,6]],axis=1)[:,None]
        y_max = np.max(gt[:,[1,3,5,7]],axis=1)[:,None]
        gt = np.concatenate((x_min, y_min, x_max-x_min, y_max-y_min),axis=1)

    data[seq] = {'images':img_list, 'gt':gt}

with open(output_path, 'wb') as fp:
    pickle.dump(data, fp, -1)
