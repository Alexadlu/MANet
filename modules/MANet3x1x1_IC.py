import os
import scipy.io
import numpy as np
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch

import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


def append_params(params, module, prefix):
    for child in module.children():
        for k,p in child._parameters.iteritems():
            if p is None: continue
            
            if isinstance(child, nn.BatchNorm2d):
                name = prefix + '_bn_' + k
            else:
                name = prefix + '_' + k
            
            if name not in params:
                params[name] = p
            else:
                raise RuntimeError("Duplicated param name: %s" % (name))


class LRN(nn.Module):
    def __init__(self):
        super(LRN, self).__init__()

    def forward(self, x):
        #
        # x: N x C x H x W
        pad = Variable(x.data.new(x.size(0), 1, 1, x.size(2), x.size(3)).zero_())
        x_sq = (x**2).unsqueeze(dim=1)
        x_tile = torch.cat((torch.cat((x_sq,pad,pad,pad,pad),2),
                            torch.cat((pad,x_sq,pad,pad,pad),2),
                            torch.cat((pad,pad,x_sq,pad,pad),2),
                            torch.cat((pad,pad,pad,x_sq,pad),2),
                            torch.cat((pad,pad,pad,pad,x_sq),2)),1)
        x_sumsq = x_tile.sum(dim=1).squeeze(dim=1)[:,2:-2,:,:]
        x = x / ((2.+0.0001*x_sumsq)**0.75)
        return x


class MDNet(nn.Module):
    def __init__(self, model_path1=None, K=1):
        super(MDNet, self).__init__()
        self.K = K
        #****************RGB_para****************
        self.RGB_para1_3x3=nn.Sequential(OrderedDict([
                                            ('Rconv1' ,nn.Sequential(nn.Conv2d(3,96,kernel_size=3,stride=2),
                                                        
                                                        nn.ReLU(),
                                                        nn.BatchNorm2d(96),
                                                        nn.Dropout(0.5),
                                                       # LRN(),
                                                        nn.MaxPool2d(kernel_size=5,stride=2)
                                            )) ]))
        
        self.RGB_para2_1x1=nn.Sequential(OrderedDict([
                                            ('Rconv2' ,nn.Sequential(nn.Conv2d(96,256,kernel_size=1,stride=2),
                                                       
                                                       nn.ReLU(),
                                                       nn.BatchNorm2d(256),
                                                       nn.Dropout(0.5),
                                                       #LRN(),
                                                       nn.MaxPool2d(kernel_size=5,stride=2))
                                            
                                            ) ]))
        
        self.RGB_para3_1x1=nn.Sequential(OrderedDict([
                                            ('Rconv3',nn.Sequential(nn.Conv2d(256,512,kernel_size=1,stride=2),
                                                      
                                                      nn.ReLU(),
                                                      nn.BatchNorm2d(512),
                                                      nn.Dropout(0.5),
                                                      #LRN()
                                                        )
                                            
                                            )]))

        
        
        #*********T_para**********************
        self.T_para1_3x3=nn.Sequential(OrderedDict([
                                            ('Tconv1' ,nn.Sequential(nn.Conv2d(3,96,kernel_size=3,stride=2),
                                                        
                                                        nn.ReLU(),
                                                        nn.BatchNorm2d(96),
                                                        nn.Dropout(0.5),
                                                        #LRN(),
                                                        nn.MaxPool2d(kernel_size=5,stride=2))
                                            ) ]))
        
        self.T_para2_1x1=nn.Sequential(OrderedDict([
                                            ('Tconv2' ,nn.Sequential(nn.Conv2d(96,256,kernel_size=1,stride=2),
                                                       
                                                       nn.ReLU(),
                                                       nn.BatchNorm2d(256),
                                                       nn.Dropout(0.5),
                                                       #LRN(),
                                                       nn.MaxPool2d(kernel_size=5,stride=2))
                                            
                                            ) ]))
        
        self.T_para3_1x1=nn.Sequential(OrderedDict([
                                            ('Tconv3',nn.Sequential(nn.Conv2d(256,512,kernel_size=1,stride=2),
                                                     
                                                      nn.ReLU(),
                                                      nn.BatchNorm2d(512),
                                                      nn.Dropout(0.5),
                                                      #LRN()
                                                      )  
                                            
                                            )]))


        self.layers = nn.Sequential(OrderedDict([
                ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                        nn.ReLU(),
                                        LRN(),
                                        nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2),
                                        nn.ReLU(),
                                        LRN(),
                                        nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1),
                                        nn.ReLU())),
                ('fc4',   nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(1024 * 3 * 3, 512),
                                        nn.ReLU())),
                ('fc5',   nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(512, 512),
                                        nn.ReLU()))]))
        
        self.branches = nn.ModuleList([nn.Sequential(nn.Dropout(0.5), 
                                                     nn.Linear(512, 2)) for _ in range(K)])
        


        if model_path1 is not None:
            if os.path.splitext(model_path1)[1] == '.pth':
                self.load_model(model_path1)
            elif os.path.splitext(model_path1)[1] == '.mat':
                self.load_mat_model(model_path1)
            else:
                raise RuntimeError("Unkown model format: %s" % (model_path1))
        self.build_param_dict()

    def build_param_dict(self):
        self.params = OrderedDict()
      
        #**********************RGB*************************************
        for name, module in self.RGB_para1_3x3.named_children():
            append_params(self.params, module, name)

        for name, module in self.RGB_para2_1x1.named_children():
            append_params(self.params, module, name)

        for name, module in self.RGB_para3_1x1.named_children():
            append_params(self.params, module, name)

        #**********************T*************************************
        for name, module in self.T_para1_3x3.named_children():
            append_params(self.params, module, name)

        for name, module in self.T_para2_1x1.named_children():
            append_params(self.params, module, name)

        for name, module in self.T_para3_1x1.named_children():
            append_params(self.params, module, name)

                                                                         
     

        #**********************conv*fc*************************************
        for name, module in self.layers.named_children():
            append_params(self.params, module, name)
        for k, module in enumerate(self.branches):
            append_params(self.params, module, 'fc6_%d'%(k))

    def set_learnable_params(self, layers):
        for k, p in self.params.iteritems():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:
                p.requires_grad = False
 
    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.params.iteritems():
            if p.requires_grad:
                params[k] = p
        return params
   

    def forward(self,xR=None, xT=None,feat=None,k=0, in_layer='conv1', out_layer='fc6'):
        
        run = False
        
        for name, module in self.layers.named_children():     
            if name == in_layer:
                run = True           
            if run:
                if name=='conv1':

                    feat_T=self.T_para1_3x3(xT)
                    feat_R=self.RGB_para1_3x3(xR)
                    feat_MT = module(xT)
                    feat_MR = module(xR)
                    
                    featT=feat_MT+feat_T
                    featR=feat_MR+feat_R
                
                
                if name=='conv2':
                    feat_T=self.T_para2_1x1(featT)
                    feat_R=self.RGB_para2_1x1(featR)
                    feat_MT=module(featT)
                    feat_MR=module(featR)

                    featR=feat_MR+feat_R
                    featT=feat_MT+feat_T

             

                if name == 'conv3':
                    feat_T=self.T_para3_1x1(featT)
                    feat_R=self.RGB_para3_1x1(featR)                   
                    feat_MT=module(featT)
                    feat_MR=module(featR)

                    featR=feat_MR+feat_R
                    featT=feat_MT+feat_T
                    feat=torch.cat((featR,featT),1)   #train 1:1   #test 1:1.4
                   
                    feat = feat.view(feat.size(0),-1)
                if name=='fc4':
                    feat=module(feat)

                if name=='fc5':
                    feat=module(feat)   

                if name == out_layer:
                    return feat
        
        feat = self.branches[k](feat)
        if out_layer=='fc6':
            return feat
        elif out_layer=='fc6_softmax':
            return F.softmax(feat,dim=1)





    
    def load_model(self, model_path):
        states = torch.load(model_path)
        shared_layers = states['shared_layers']
        self.layers.load_state_dict(shared_layers)

        
        para1_layers=states['RGB_para1_3x3']
        self.RGB_para1_3x3.load_state_dict(para1_layers,strict=True)
        para2_layers=states['RGB_para2_1x1']
        self.RGB_para2_1x1.load_state_dict(para2_layers,strict=True)
        para3_layers=states['RGB_para3_1x1']
        self.RGB_para3_1x1.load_state_dict(para3_layers,strict=True)

        
        para1_layers=states['T_para1_3x3']
        self.T_para1_3x3.load_state_dict(para1_layers,strict=True)
        para2_layers=states['T_para2_1x1']
        self.T_para2_1x1.load_state_dict(para2_layers,strict=True)
        para3_layers=states['T_para3_1x1']
        self.T_para3_1x1.load_state_dict(para3_layers,strict=True)

       
        
        print('load finish pth!!!')
        
        

    def load_mat_model(self, matfile):
        mat = scipy.io.loadmat(matfile)
        mat_layers = list(mat['layers'])[0]
        
        # copy conv weights
        for i in range(3):
            weight, bias = mat_layers[i*4]['weights'].item()[0]
            self.layers[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3,2,0,1)))
            self.layers[i][0].bias.data = torch.from_numpy(bias[:,0])

        print('load mat finish!')

class BinaryLoss(nn.Module):
    def __init__(self):
        super(BinaryLoss, self).__init__()
 
    def forward(self, pos_score, neg_score):
        pos_loss = -F.log_softmax(pos_score,dim=1)[:,1]
        neg_loss = -F.log_softmax(neg_score,dim=1)[:,0]
        
        loss = pos_loss.sum() + neg_loss.sum()
        return loss


class Accuracy():
    def __call__(self, pos_score, neg_score):
        
        pos_correct = (pos_score[:,1] > pos_score[:,0]).sum().float()
        neg_correct = (neg_score[:,1] < neg_score[:,0]).sum().float()
        
        pos_acc = pos_correct / (pos_score.size(0) + 1e-8)
        neg_acc = neg_correct / (neg_score.size(0) + 1e-8)

        return pos_acc.data[0], neg_acc.data[0]


class Precision():
    def __call__(self, pos_score, neg_score):
        
        scores = torch.cat((pos_score[:,1], neg_score[:,1]), 0)
        topk = torch.topk(scores, pos_score.size(0))[1]
        prec = (topk < pos_score.size(0)).float().sum() / (pos_score.size(0)+1e-8)
        
        return prec.data[0]
