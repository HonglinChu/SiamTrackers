# -*- coding: utf-8 -*
from loguru import logger

import torch
import torch.nn as nn

from siamfcpp.model.backbone.backbone_base import (TRACK_BACKBONES,
                                                       VOS_BACKBONES)
from siamfcpp.model.common_opr.common_block import conv_bn_relu
from siamfcpp.model.module_base import ModuleBase
from siamfcpp.utils import md5sum


@VOS_BACKBONES.register
@TRACK_BACKBONES.register
class TinyConv(ModuleBase):
    r"""
    TinyNet
    Customized, extremely pruned ConvNet

    Hyper-parameters
    ----------------
    pretrain_model_path: string
        Path to pretrained backbone parameter file,
        Parameter to be loaded in _update_params_
    """
    default_hyper_params = {"pretrain_model_path": ""}

    def __init__(self):
        super(TinyConv, self).__init__()

        self.conv1 = conv_bn_relu(3, 32, stride=2, kszie=3, pad=0)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=0, ceil_mode=True)

        self.conv2a = conv_bn_relu(32, 64, stride=1, kszie=1, pad=0)
        self.conv2b = conv_bn_relu(64, 64, stride=2, kszie=7, pad=0, groups=64)

        self.conv3a = conv_bn_relu(64, 64, stride=1, kszie=3, pad=0)
        self.conv3b = conv_bn_relu(64,
                                   64,
                                   stride=1,
                                   kszie=1,
                                   pad=0,
                                   has_relu=False)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.as_tensor(X.rvs(m.weight.numel()),
                                         dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2a(x)
        x = self.conv2b(x)

        x = self.conv3a(x)
        x = self.conv3b(x)

        return x
