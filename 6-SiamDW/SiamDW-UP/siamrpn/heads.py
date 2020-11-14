import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from .transforms import ToTensor

from torchvision.models import alexnet
from torch.autograd import Variable
from torch import nn

from IPython import embed
from .config import config

from .backbones import *

__all__=['SiamRPN']

class SiamRPN(nn.Module):

    def __init__(self):

        super(SiamRPN, self).__init__()
        
        self.channel=512


        self.anchor_num = config.anchor_num   #每一个位置有5个anchor

        self.input_size = config.instance_size

        self.score_displacement = int((self.input_size - config.exemplar_size) / config.total_stride)

        self.conv_cls1 = nn.Conv2d(self.channel, self.channel * 2 * self.anchor_num, kernel_size=3, stride=1, padding=0)

        self.conv_r1 = nn.Conv2d(self.channel, self.channel * 4 * self.anchor_num, kernel_size=3, stride=1, padding=0)

        self.conv_cls2 = nn.Conv2d(self.channel, self.channel, kernel_size=3, stride=1, padding=0)

        self.conv_r2 = nn.Conv2d(self.channel, self.channel, kernel_size=3, stride=1, padding=0)

        self.regress_adjust = nn.Conv2d(4 * self.anchor_num, 4 * self.anchor_num, 1)# 1x1的卷积？

    def forward(self, z, x):

        N=z.size(0)

        kernel_score = self.conv_cls1(z).view(N, 2 * self.anchor_num, self.channel,3, 3)    #32,2*5,self.channel,4,4
        
        kernel_regression = self.conv_r1(z).view(N, 4 * self.anchor_num, self.channel, 3, 3) #32,4*5,self.channel,4,4
        
        conv_score = self.conv_cls2(x) #32,self.channel,22,22#对齐操作
        
        conv_regression = self.conv_r2(x)   # 32, self.channel , 22, 22
        #conv_scores = conv_score.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)#1,32xself.channel,22,22
        conv_scores = conv_score.reshape(1, -1, self.score_displacement + 3, self.score_displacement + 3)#1,32xself.channel,22,22
        # 1，8192,22,22
        score_filters = kernel_score.reshape(-1, self.channel, 3, 3) # 32x10,self.channel,4,4
       
        pred_score = F.conv2d(conv_scores, score_filters, groups=N).reshape(N, 10, self.score_displacement + 1,self.score_displacement + 1)
        #32,10,19,19
        conv_reg = conv_regression.reshape(1, -1, self.score_displacement + 3, self.score_displacement + 3)
        #32,self.channel,22,22
        reg_filters = kernel_regression.reshape(-1, self.channel, 3,3)
        #640,self.channel, 4, 4 
        pred_regression = self.regress_adjust(F.conv2d(conv_reg, reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1,self.score_displacement + 1))
        #32, 20, 19, 19
        return pred_score, pred_regression
    
    def z_branch(self, z):
        N=z.size(0)
        kernel_score = self.conv_cls1(z).view(N, 2 * self.anchor_num, self.channel, 3, 3)
        kernel_regression = self.conv_r1(z).view(N, 4 * self.anchor_num, self.channel, 3, 3)
        self.score_filters = kernel_score.reshape(-1, self.channel, 3, 3)   # 2x5, self.channel, 4, 4  
        self.reg_filters = kernel_regression.reshape(-1, self.channel, 3, 3)# 4x5, self.channel, 4, 4

    def x_branch(self, x):

        N=x.size(0)

        conv_score = self.conv_cls2(x) #输入通道self.channel，输出通道self.channel，kernel=3，stride=1，padding[1,self.channel,20,20]
       
        conv_regression = self.conv_r2(x)#输入通道self.channel，输出通道self.channel，kernel=3，stride=1，padding[1,self.channel,20,20]
        
        conv_scores = conv_score.reshape(1, -1, self.score_displacement + 3, self.score_displacement + 3)#????
        
        pred_score = F.conv2d(conv_scores, self.score_filters, groups=N).reshape(N, 10, self.score_displacement + 1,
                                                                                 self.score_displacement + 1)
        conv_reg = conv_regression.reshape(1, -1, self.score_displacement + 3, self.score_displacement + 3)
        
        pred_regression = self.regress_adjust(
           
            F.conv2d(conv_reg, self.reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1,self.score_displacement + 1))
        
        return pred_score, pred_regression 