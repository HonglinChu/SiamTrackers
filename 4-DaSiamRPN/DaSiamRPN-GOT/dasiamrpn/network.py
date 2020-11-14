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

class SiamRPNNet(nn.Module):
    def __init__(self, size=2, feature_out=512):
        super(SiamRPNNet, self).__init__()

        configs = [3, 96, 256, 384, 384, 256]
        configs = list(map(lambda x: 3 if x==3 else x*size, configs))
        feat_in = configs[-1]

        self.featureExtract = nn.Sequential(

            nn.Conv2d(configs[0], configs[1] , kernel_size=11, stride=2),
            nn.BatchNorm2d(configs[1]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[1], configs[2], kernel_size=5),
            nn.BatchNorm2d(configs[2]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[2], configs[3], kernel_size=3),
            nn.BatchNorm2d(configs[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[3], configs[4], kernel_size=3),
            nn.BatchNorm2d(configs[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[4], configs[5], kernel_size=3),
            nn.BatchNorm2d(configs[5]),
        )
        self.anchor_num = config.anchor_num   #每一个位置有5个anchor

        self.input_size = config.instance_size

        self.feature_out = feature_out

        self.score_displacement = int((self.input_size - config.exemplar_size) / config.total_stride)
        self.conv_cls1 = nn.Conv2d(feat_in, feature_out * 2 * self.anchor_num, kernel_size=3)
        self.conv_r1 = nn.Conv2d(feat_in, feature_out * 4 * self.anchor_num, kernel_size=3)

        self.conv_cls2 = nn.Conv2d(feat_in,feature_out, kernel_size=3)
        self.conv_r2 = nn.Conv2d(feat_in, feature_out, kernel_size=3)
        self.regress_adjust = nn.Conv2d(4 * self.anchor_num, 4 * self.anchor_num, 1)# 1x1的卷积？


    def forward(self, template, detection):
        
        N = template.size(0) # N=32
       
        template_feature = self.featureExtract(template)#[32,256,6,6]
        
        detection_feature = self.featureExtract(detection)#[32,256,24,24]

        kernel_score = self.conv_cls1(template_feature).view(N, 2 * self.anchor_num,  self.feature_out, 4, 4)    #32,2*5,256,4,4
        
        kernel_regression = self.conv_r1(template_feature).view(N, 4 * self.anchor_num,  self.feature_out, 4, 4) #32,4*5,256,4,4
       
        conv_score = self.conv_cls2(detection_feature) #32,256,22,22#对齐操作

        conv_regression = self.conv_r2(detection_feature)#32,256,22,22

        conv_scores = conv_score.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)#1,32x256,22,22
        
        # 1，8192,22,22
        score_filters = kernel_score.reshape(-1,  self.feature_out, 4, 4) # 32x10,256,4,4
       
        pred_score = F.conv2d(conv_scores, score_filters, groups=N).reshape(N, 10, self.score_displacement + 1,self.score_displacement + 1)
        #32,10,19,19
        conv_reg = conv_regression.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        #32,256,22,22
        reg_filters = kernel_regression.reshape(-1,  self.feature_out, 4, 4)
        #640,256, 4, 4
        pred_regression = self.regress_adjust(F.conv2d(conv_reg, reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1,self.score_displacement + 1))
        #32, 20, 19, 19
        return pred_score, pred_regression


    def track_init(self, template):
        N = template.size(0)
        template_feature = self.featureExtract(template)# 输出 [1, 256, 6, 6]
        # kernel_score=1,2x5,256,4,4   kernel_regression=1,4x5, 256,4,4
        kernel_score = self.conv_cls1(template_feature).view(N, 2 * self.anchor_num,  self.feature_out, 4, 4)
        kernel_regression = self.conv_r1(template_feature).view(N, 4 * self.anchor_num,  self.feature_out, 4, 4)
        self.score_filters = kernel_score.reshape(-1,  self.feature_out, 4, 4)   # 2x5, 256, 4, 4  
        self.reg_filters = kernel_regression.reshape(-1,  self.feature_out, 4, 4)# 4x5, 256, 4, 4

    def track(self, detection):
        N = detection.size(0)
        detection_feature = self.featureExtract(detection)# 1,256,22,22
        
        conv_score = self.conv_cls2(detection_feature) #输入通道256，输出通道256，kernel=3，stride=1，padding[1,256,20,20]
        conv_regression = self.conv_r2(detection_feature)#输入通道256，输出通道256，kernel=3，stride=1，padding[1,256,20,20]
        
        conv_scores = conv_score.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)#????
        pred_score = F.conv2d(conv_scores, self.score_filters, groups=N).reshape(N, 10, self.score_displacement + 1,
                                                                                 self.score_displacement + 1)
        conv_reg = conv_regression.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        pred_regression = self.regress_adjust(
            F.conv2d(conv_reg, self.reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1,self.score_displacement + 1))
        return pred_score, pred_regression

