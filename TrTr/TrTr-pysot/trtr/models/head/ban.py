from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import math  
from trtr.core.xcorr import xcorr_fast, xcorr_depthwise
from trtr.core.config import cfg
from trtr.models.head.trans_head  import Transformer 

class BAN(nn.Module):
    def __init__(self):
        super(BAN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError

class UPChannelBAN(BAN):
    def __init__(self, feature_in=256, cls_out_channels=2):
        super(UPChannelBAN, self).__init__()

        cls_output = cls_out_channels
        loc_output = 4

        self.template_cls_conv = nn.Conv2d(feature_in, 
                feature_in * cls_output, kernel_size=3)
        self.template_loc_conv = nn.Conv2d(feature_in, 
                feature_in * loc_output, kernel_size=3)

        self.search_cls_conv = nn.Conv2d(feature_in, 
                feature_in, kernel_size=3)
        self.search_loc_conv = nn.Conv2d(feature_in, 
                feature_in, kernel_size=3)

        self.loc_adjust = nn.Conv2d(loc_output, loc_output, kernel_size=1)


    def forward(self, z_f, x_f):
        cls_kernel = self.template_cls_conv(z_f)
        loc_kernel = self.template_loc_conv(z_f)

        cls_feature = self.search_cls_conv(x_f)
        loc_feature = self.search_loc_conv(x_f)

        cls = xcorr_fast(cls_feature, cls_kernel)
        loc = self.loc_adjust(xcorr_fast(loc_feature, loc_kernel))
        return cls, loc


class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels,  out_channels, kernel_size=3):
        super(DepthwiseXCorr, self).__init__()

        self.conv_kernel = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                )
        self.conv_search = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                )
        
        for modules in [self.conv_kernel, self.conv_search]:
            for m in modules.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel) #[64,96,2,2]
        search = self.conv_search(search) #[64,96,14,14]
        feature = xcorr_depthwise(search, kernel)
        return feature 

class Bottleneck(nn.Module):
    
    def __init__(self,
                 in_channels: int = 96,
                 mid_channels: int = 96,
                 dilation: int = 1):
        super(Bottleneck, self).__init__()

        self.in_channels=in_channels
        self.mid_channels=mid_channels

        self.pw1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0,bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU6(inplace=True)
        )  

        self.dw = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels,kernel_size=3, padding=1,groups=mid_channels,bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU6(inplace=True)
        )
        
        self.pw2 = nn.Sequential(
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            ##nn.ReLU6(inplace=True) # YOLOF have relu-->  MobileNetV2 no relu 
        )

    def forward(self, x): 
        identity = x 
        # pw
        if  self.mid_channels!=self.in_channels:
            out= self.pw1(x)
            # dw
            out = self.dw(out)
            # pw
            out = self.pw2(out)
        else:
            # dw
            out = self.dw(x)
            # pw
            out = self.pw2(out)

        # shortcut
        out = out + identity

        return out 

class DepthwiseBAN(BAN):
    def __init__(self, in_channels=96, out_channels=96, weighted=False):
        super(DepthwiseBAN, self).__init__()

        self.cls_dw = DepthwiseXCorr(in_channels, out_channels)
        
        self.reg_dw = DepthwiseXCorr(in_channels, out_channels)

        cls_tower = []
        bbox_tower = [] 

        # （1）特征增强网络 4
        for i in range(cfg.TRAIN.NUM_CONVS):  # 
            cls_tower.append(nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ))
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())

            bbox_tower.append(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ))
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        # add_module 函数，为module添加一个子module函数  https://blog.csdn.net/qq_31964037/article/details/105416291
        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))

        self.cls_logits = nn.Conv2d(
            in_channels, 2, kernel_size=3, stride=1,
            padding=1
        ) 

        # （3）回归分支输出  input:[256] --> output:[4]
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )

        self.trans_head = Transformer(cfg)

        #  （5）权重初始化 在这里只是对conv分支进行初始化？？？initialization
        for modules in [self.cls_tower, self.bbox_tower, self.trans_head,
                        self.cls_logits, self.bbox_pred]:
            for m in modules.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()

    def forward(self, z_f, x_f):

        # x_cls = self.cls_dw(z_f, x_f) 

        # x_reg = self.reg_dw(z_f, x_f) 

        zxf_cls=self.trans_head(z_f,x_f) # [1,16,256, 96]

        zxf_cls=zxf_cls.transpose(2,3).squeeze(0) 
        B,C,WH=zxf_cls.shape
        W,H=int(math.sqrt(WH)),int(math.sqrt(WH))
        zxf_cls=zxf_cls.view(B,C,H,W) #
        
        # head-分类
        cls_tower = self.cls_tower(zxf_cls) #[B, 256, 25, 25] --> [B, 256, 25, 25]

        logits = self.cls_logits(cls_tower) # [B, 256, 25, 25] --> [B, 2, 25, 25]

        # head-回归
        bbox_tower=self.bbox_tower(zxf_cls)

        bbox_reg=self.bbox_pred(bbox_tower) #[B, 256, 25, 25] --> [B, 4, 25, 25]
        
        bbox_reg = torch.exp(bbox_reg)  #  siamban  siamcar  [B, 4, 25, 25] --> [B, 4, 25, 25]

        return logits, bbox_reg 

class MultiBAN(BAN):
    def __init__(self, in_channels, cls_out_channels, weighted=False):
        super(MultiBAN, self).__init__()
        self.weighted = weighted
        for i in range(len(in_channels)):
            self.add_module('box'+str(i+2), DepthwiseBAN(in_channels[i], in_channels[i], cls_out_channels))
        if self.weighted:
            self.cls_weight = nn.Parameter(torch.ones(len(in_channels)))
            self.loc_weight = nn.Parameter(torch.ones(len(in_channels)))
        self.loc_scale = nn.Parameter(torch.ones(len(in_channels)))

    def forward(self, z_fs, x_fs):
        cls = []
        loc = []
        for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs), start=2):
            box = getattr(self, 'box'+str(idx))
            c, l = box(z_f, x_f)
            cls.append(c)
            loc.append(torch.exp(l*self.loc_scale[idx-2]))

        if self.weighted:
            cls_weight = F.softmax(self.cls_weight, 0)
            loc_weight = F.softmax(self.loc_weight, 0)

        def avg(lst):
            return sum(lst) / len(lst)

        def weighted_avg(lst, weight):
            s = 0
            for i in range(len(weight)):
                s += lst[i] * weight[i]
            return s

        if self.weighted:
            return weighted_avg(cls, cls_weight), weighted_avg(loc, loc_weight)
        else:
            return avg(cls), avg(loc)
