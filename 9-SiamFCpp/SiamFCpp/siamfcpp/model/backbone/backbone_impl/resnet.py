# -*- coding: utf-8 -*

from loguru import logger

import torch
import torch.nn as nn

from siamfcpp.model.backbone.backbone_base import (TRACK_BACKBONES,
                                                       VOS_BACKBONES)
from siamfcpp.model.common_opr.common_block import conv_bn_relu, projector
from siamfcpp.model.module_base import ModuleBase


class creat_residual_block(nn.Module):
    def __init__(self, inplanes, outplanes, stride, has_proj=False):
        super(creat_residual_block, self).__init__()
        self.has_proj = has_proj
        if self.has_proj:
            self.proj_conv = conv_bn_relu(inplanes,
                                          outplanes,
                                          stride=stride,
                                          kszie=1,
                                          pad=0,
                                          has_bn=True,
                                          has_relu=False,
                                          bias=False)

        self.conv1 = conv_bn_relu(inplanes,
                                  outplanes,
                                  stride=stride,
                                  kszie=3,
                                  pad=1,
                                  has_bn=True,
                                  has_relu=True,
                                  bias=False)
        self.conv2 = conv_bn_relu(outplanes,
                                  outplanes,
                                  stride=1,
                                  kszie=3,
                                  pad=1,
                                  has_bn=True,
                                  has_relu=False,
                                  bias=False)

        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        if self.has_proj:
            residual = self.proj_conv(residual)

        x = self.conv1(x)
        x = self.conv2(x)
        x = x + residual
        x = self.relu(x)
        return x


class create_bottleneck(nn.Module):
    """
    Modified Bottleneck : We change the kernel size of projection conv from 1 to 3.

    """
    def __init__(self, inplanes, outplanes, stride, has_proj=False):
        super(create_bottleneck, self).__init__()
        self.has_proj = has_proj
        if self.has_proj:
            self.proj_conv = conv_bn_relu(inplanes,
                                          outplanes,
                                          stride=stride,
                                          kszie=3,
                                          pad=1,
                                          has_bn=True,
                                          has_relu=False,
                                          bias=False)

        self.conv1 = conv_bn_relu(inplanes,
                                  outplanes,
                                  stride=stride,
                                  kszie=3,
                                  pad=1,
                                  has_bn=True,
                                  has_relu=True,
                                  bias=False)
        self.conv2 = conv_bn_relu(outplanes,
                                  outplanes,
                                  stride=1,
                                  kszie=3,
                                  pad=1,
                                  has_bn=True,
                                  has_relu=True,
                                  bias=False)
        self.conv3 = conv_bn_relu(outplanes,
                                  outplanes,
                                  stride=1,
                                  kszie=3,
                                  pad=1,
                                  has_bn=True,
                                  has_relu=False,
                                  bias=False)

        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        if self.has_proj:
            residual = self.proj_conv(residual)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + residual
        x = self.relu(x)
        return x


@VOS_BACKBONES.register
class ResNet50_M(ModuleBase):

    default_hyper_params = {"pretrain_model_path": ""}

    def __init__(self, block=create_bottleneck):
        super(ResNet50_M, self).__init__()
        self.block = block
        self.stage1 = nn.Sequential(
            conv_bn_relu(3,
                         32,
                         stride=2,
                         kszie=3,
                         pad=3,
                         has_bn=True,
                         has_relu=True,
                         bias=False),
            conv_bn_relu(32,
                         32,
                         stride=1,
                         kszie=3,
                         pad=1,
                         has_bn=True,
                         has_relu=True,
                         bias=False),
            conv_bn_relu(32,
                         32,
                         stride=1,
                         kszie=3,
                         pad=1,
                         has_bn=True,
                         has_relu=True,
                         bias=False), nn.MaxPool2d(3, 2, 1, ceil_mode=False))
        self.stage2 = self.__make_stage(self.block, 32, 64, 3, 1)
        self.stage3 = self.__make_stage(self.block, 64, 128, 4, 2)
        self.stage4 = self.__make_stage(self.block, 128, 256, 6, 2)
        self.stage5 = self.__make_stage(self.block, 256, 512, 3, 2)

    def __make_stage(self, block, inplane, outplane, blocks, stride):
        layers = []
        layers.append(block(inplane, outplane, stride=stride, has_proj=True))
        for i in range(1, blocks):
            layers.append(block(outplane, outplane, 1, False))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        return x5


@VOS_BACKBONES.register
class ResNet18_M(ModuleBase):

    default_hyper_params = {"pretrain_model_path": ""}

    def __init__(self, block=creat_residual_block):
        super(ResNet18_M, self).__init__()
        self.block = block
        self.stage1 = nn.Sequential(
            conv_bn_relu(3,
                         32,
                         stride=2,
                         kszie=3,
                         pad=3,
                         has_bn=True,
                         has_relu=True,
                         bias=False),
            conv_bn_relu(32,
                         32,
                         stride=1,
                         kszie=3,
                         pad=1,
                         has_bn=True,
                         has_relu=True,
                         bias=False),
            conv_bn_relu(32,
                         32,
                         stride=1,
                         kszie=3,
                         pad=1,
                         has_bn=True,
                         has_relu=True,
                         bias=False), nn.MaxPool2d(3, 2, 1, ceil_mode=False))
        self.stage2 = self.__make_stage(self.block, 32, 64, 2, 1)
        self.stage3 = self.__make_stage(self.block, 64, 128, 2, 2)
        self.stage4 = self.__make_stage(self.block, 128, 256, 2, 2)
        self.stage5 = self.__make_stage(self.block, 256, 256, 2, 2)

    def __make_stage(self, block, inplane, outplane, blocks, stride):
        layers = []
        layers.append(block(inplane, outplane, stride=stride, has_proj=True))
        for i in range(1, blocks):
            layers.append(block(outplane, outplane, 1, False))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        return x5


@VOS_BACKBONES.register
class JointEncoder(ModuleBase):

    default_hyper_params = {"pretrain_model_path": ""}

    def __init__(self, basemodel):
        super(JointEncoder, self).__init__()
        self.basemodel = basemodel
        self.projector_corr_feature = projector(256, 256)

    def forward(self, saliency_image, corr_feature):
        corr_feature = self.projector_corr_feature(corr_feature)
        x1 = self.basemodel.stage1(saliency_image)
        x2 = self.basemodel.stage2(x1)
        x3 = self.basemodel.stage3(x2)
        x4 = self.basemodel.stage4(x3) + corr_feature
        x5 = self.basemodel.stage5(x4)
        return [x5, x4, x3, x2]


if __name__ == "__main__":
    print(VOS_BACKBONES)
    resnet_m = ResNet18_M()
    image = torch.rand((1, 3, 257, 257))
    print(image.shape)
    feature = resnet_m(image)
    print(feature.shape)
    print(resnet_m.state_dict().keys())
    #print(resnet_m)
