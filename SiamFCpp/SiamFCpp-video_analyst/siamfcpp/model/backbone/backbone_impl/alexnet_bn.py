# -*- coding: utf-8 -*
from loguru import logger

import torch
import torch.nn as nn

from siamfcpp.model.backbone.backbone_base import (TRACK_BACKBONES,
                                                       VOS_BACKBONES)
from siamfcpp.model.common_opr.common_block import conv_bn_relu
from siamfcpp.model.module_base import ModuleBase

#Registry装饰器类来实现构造不同backbone结构函数的调用
#定义不同的backbone并将其保存在TRACK_BACKBONES 和 VOS_BACKBONES 型字典中
@VOS_BACKBONES.register
@TRACK_BACKBONES.register
class AlexNet(ModuleBase):

    r"""
    AlexNet

    Hyper-parameters
    ----------------
    pretrain_model_path: string
        Path to pretrained backbone parameter file,
        Parameter to be loaded in _update_params_
    """
    default_hyper_params = {"pretrain_model_path": ""}

    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = conv_bn_relu(3, 96, stride=2, kszie=11, pad=0)
        self.pool1 = nn.MaxPool2d(3, 2, 0, ceil_mode=True)
        self.conv2 = conv_bn_relu(96, 256, 1, 5, 0)
        self.pool2 = nn.MaxPool2d(3, 2, 0, ceil_mode=True)
        self.conv3 = conv_bn_relu(256, 384, 1, 3, 0)
        self.conv4 = conv_bn_relu(384, 384, 1, 3, 0)
        self.conv5 = conv_bn_relu(384, 256, 1, 3, 0, has_relu=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x
