import torch
from torch import nn
import math
from siamcar.utils.xcorr import DepthwiseXCorr


class CARHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(CARHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.TRAIN.NUM_CLASSES

        cls_tower = []
        bbox_tower = [] 

        # （1）特征增强网络
        # 组归一化，避开了batch size对模型的影响，在channel方向分group，然后每个group内做归一化 https://blog.csdn.net/duanshao/article/details/80055887
        # cls分支 4层：【卷积 -- 组归一化 -- 激活】，【卷积 -- 组归一化 -- 激活】，【卷积 -- 组归一化 -- 激活】，【卷积 -- 组归一化 -- 激活】
        # reg分支 4层：【卷积 -- 组归一化 -- 激活】，【卷积 -- 组归一化 -- 激活】，【卷积 -- 组归一化 -- 激活】，【卷积 -- 组归一化 -- 激活】
        for i in range(cfg.TRAIN.NUM_CONVS):  # 
            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())
        # add_module 函数，为module添加一个子module函数  https://blog.csdn.net/qq_31964037/article/details/105416291
        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))

        # （2）分类分支输出  input:[256] --> output:[2]
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        ) 

        # （3）回归分支输出  input:[256] --> output:[4]
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )

        # （4）中心分支输出  input:[256] --> output:[1]
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )
        
        out_channels=in_channels
        self.cls_dw = DepthwiseXCorr(in_channels, out_channels)
        self.reg_dw = DepthwiseXCorr(in_channels, out_channels)


        #  （5）权重初始化 在这里只是对conv分支进行初始化？？？initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # （6） ？？？ initialize the bias for focal loss
        prior_prob = cfg.TRAIN.PRIOR_PROB # 0.01 
        #  -4.595 ？？？ 
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        torch.nn.init.constant_(self.cls_logits.bias, bias_value)
        
    def forward(self, z_f,x_f):

        # 两个分支分别进行 dw 相关操作
        x_cls=self.cls_dw(z_f,x_f)

        x_reg=self.reg_dw(z_f,x_f)

        # head-分类
        cls_tower = self.cls_tower(x_cls) #[B, 256, 25, 25] --> [B, 256, 25, 25]

        logits = self.cls_logits(cls_tower) # [B, 256, 25, 25] --> [B, 2, 25, 25]

        centerness = self.centerness(cls_tower) #[B, 256, 25, 25] --> [B, 1, 25, 25]

        # head-回归
        bbox_tower=self.bbox_tower(x_reg)

        bbox_reg=self.bbox_pred(bbox_tower) #[B, 256, 25, 25] --> [B, 4, 25, 25]
        
        bbox_reg = torch.exp(bbox_reg) # [B, 4, 25, 25] --> [B, 4, 25, 25]

        return logits, bbox_reg, centerness 

class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale
