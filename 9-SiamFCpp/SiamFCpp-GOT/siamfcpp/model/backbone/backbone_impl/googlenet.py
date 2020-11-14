# -*- coding: utf-8 -*
"""
Adapted from the implementation @ PyTorch model zoo
URL: https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
Pretrained weights downloaded from:
    https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth
"""
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from siamfcpp.model.backbone.backbone_base import (TRACK_BACKBONES,
                                                       VOS_BACKBONES)
from siamfcpp.model.module_base import ModuleBase

# from siamfcpp.model.common_opr.common_block import conv_bn_relu


@VOS_BACKBONES.register
@TRACK_BACKBONES.register
class Inception3(ModuleBase):
    r"""
    GoogLeNet

    Hyper-parameters
    ----------------
    pretrain_model_path: string
        Path to pretrained backbone parameter file,
        Parameter to be loaded in _update_params_
    crop_pad: int
        width of pixels to be cropped at each edge
    pruned: bool
        if using pruned backbone for SOT
    """
    default_hyper_params = dict(
        pretrain_model_path="",
        crop_pad=4,
        pruned=True,
    )

    def __init__(self, transform_input=False):
        super(Inception3, self).__init__()
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        # The last stage is not used in our experiment, resulting in faster inference speed.
        # self.Mixed_7a = InceptionD(768)
        # self.Mixed_7b = InceptionE(1280)
        # self.Mixed_7c = InceptionE(2048)
        # self.fc = nn.Linear(2048, num_classes)

        # Parameters are loaded, no need to initialized
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #         import scipy.stats as stats
        #         stddev = m.stddev if hasattr(m, 'stddev') else 0.1
        #         X = stats.truncnorm(-2, 2, scale=stddev)
        #         values = torch.as_tensor(X.rvs(m.weight.numel()),
        #                                  dtype=m.weight.dtype)
        #         values = values.view(m.weight.size())
        #         with torch.no_grad():
        #             m.weight.copy_(values)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

        self.channel_reduce = nn.Sequential(
            nn.Conv2d(768, 256, 1),
            nn.BatchNorm2d(256, eps=0.001),
        )

    def forward(self, x):
        # RGB -> BGR, [0, 255] -> [-1, 1]
        bias = 255 / 2
        x_ch0 = (torch.unsqueeze(x[:, 2], 1) - bias) / bias
        x_ch1 = (torch.unsqueeze(x[:, 1], 1) - bias) / bias
        x_ch2 = (torch.unsqueeze(x[:, 0], 1) - bias) / bias
        x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        # max_pool2d pruned for SOT adapdation
        # x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17

        # cropping to alleviate
        crop_pad = self.crop_pad
        x = x[:, :, crop_pad:-crop_pad, crop_pad:-crop_pad]
        x = self.channel_reduce(x)
        return x

        # The last stage is not used in our experiment, resulting in faster inference speed.
        # # N x 768 x 17 x 17
        # x = self.Mixed_7a(x)
        # # N x 1280 x 8 x 8
        # x = self.Mixed_7b(x)
        # # N x 2048 x 8 x 8
        # x = self.Mixed_7c(x)
        # # N x 2048 x 8 x 8
        # # Adaptive average pooling
        # x = F.adaptive_avg_pool2d(x, (1, 1))
        # # N x 2048 x 1 x 1
        # x = F.dropout(x, training=self.training)
        # # N x 2048 x 1 x 1
        # x = torch.flatten(x, 1)
        # # N x 2048
        # # x = self.fc(x)
        # # N x 1000 (num_classes)
        # # if self.training and self.aux_logits:
        # #     return _InceptionOutputs(x, aux)
        # return x

    def update_params(self):
        super().update_params()
        self.crop_pad = self._hyper_params['crop_pad']
        self.pruned = self._hyper_params['pruned']


class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels,
                                       pool_features,
                                       kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):
    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):
    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7,
                                       c7,
                                       kernel_size=(1, 7),
                                       padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7,
                                       192,
                                       kernel_size=(7, 1),
                                       padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7,
                                          c7,
                                          kernel_size=(7, 1),
                                          padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7,
                                          c7,
                                          kernel_size=(1, 7),
                                          padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7,
                                          c7,
                                          kernel_size=(7, 1),
                                          padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7,
                                          192,
                                          kernel_size=(1, 7),
                                          padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):
    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192,
                                         192,
                                         kernel_size=(1, 7),
                                         padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192,
                                         192,
                                         kernel_size=(7, 1),
                                         padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):
    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384,
                                        384,
                                        kernel_size=(1, 3),
                                        padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384,
                                        384,
                                        kernel_size=(3, 1),
                                        padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384,
                                           384,
                                           kernel_size=(1, 3),
                                           padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384,
                                           384,
                                           kernel_size=(3, 1),
                                           padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # N x 768 x 17 x 17
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 768 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 768
        x = self.fc(x)
        # N x 1000
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
