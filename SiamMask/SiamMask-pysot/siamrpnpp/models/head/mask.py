# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch 
import torch.nn as nn
import torch.nn.functional as F

from siamrpnpp.models.head.rpn import DepthwiseXCorr 
from siamrpnpp.core.xcorr import xcorr_depthwise

# ----------------------------------------------SiamMask-----------------------------------------#
class ConvBlock(nn.Module):
    def __init__(self, inplane, outplane, kernel_size, padding, stride):
        super(ConvBlock, self).__init__()
        self.convmodule = nn.Sequential(
                nn.Conv2d(inplane, outplane, kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
                nn.BatchNorm2d(outplane),
                nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.convmodule(x)
        return x

# ----------------------------------------------SiamMask-----------------------------------------#
class Mask(nn.Module):
    def __init__(self):
        super(Mask, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError

    def template(self, template):
        raise NotImplementedError

    def track(self, search):
        raise NotImplementedError

    def param_groups(self, start_lr, feature_mult=1):
        params = filter(lambda x:x.requires_grad, self.parameters())
        params = [{'params': params, 'lr': start_lr * feature_mult}]
        return params

class MaskCorr(Mask):
    def __init__(self, oSz=63,weighted=False):
        super(MaskCorr, self).__init__()
        self.oSz = oSz
        self.mask = DepthwiseXCorr(256, 256, self.oSz**2)
    def forward(self, kernel, search):
        out = self.mask(kernel, search)
        return out

class MultiMaskCorr(Mask):
    def __init__(self, oSz=63, weighted=False):
        super(MultiMaskCorr, self).__init__()
        self.weighted = weighted
        for i in range(3):
            self.add_module('maskcorr' + str(i + 2),
                            MaskCorr())
        if self.weighted:
            self.mask_weight = nn.Parameter(torch.ones(3))

    def forward(self, z_fs, x_fs):
        mask = []
        for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs), start=2):
            maskcorr = getattr(self, 'maskcorr'+str(idx))
            m = maskcorr(z_f, x_f)
            mask.append(m)

        if self.weighted:
            mask_weight = F.softmax(self.mask_weight, 0)

        def avg(lst):
            return sum(lst) / len(lst)

        def weighted_avg(lst, weight):
            s = 0
            for i in range(len(weight)):
                s += lst[i] * weight[i]
            return s

        if self.weighted:
            return weighted_avg(mask, mask_weight)
        else:
            return avg(mask)

    def forward_corrs(self, z_fs, x_fs):
        mask = []
        for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs), start=2):
            maskcorr = getattr(self, 'maskcorr' + str(idx))
            m = maskcorr.mask.forward_corr(z_f, x_f)
            mask.append(m)

        if self.weighted:
            mask_weight = F.softmax(self.mask_weight, 0)

        def avg(lst):
            return sum(lst) / len(lst)

        def weighted_avg(lst, weight):
            s = 0
            for i in range(len(weight)):
                s += lst[i] * weight[i]
            return s

        if self.weighted:
            return weighted_avg(mask, mask_weight)
        else:
            return avg(mask)

# ----------------------------------------------SiamMask-----------------------------------------#
# remove follow code 
# class MaskCorr(DepthwiseXCorr):
#     def __init__(self, in_channels, hidden, out_channels,
#                  kernel_size=3, hidden_kernel_size=5):
#         super(MaskCorr, self).__init__(in_channels, hidden,
#                                        out_channels, kernel_size,
#                                        hidden_kernel_size)
#     def forward(self, kernel, search):
#         kernel = self.conv_kernel(kernel)
#         search = self.conv_search(search)
#         feature = xcorr_depthwise(search, kernel)
#         out = self.head(feature)
#         return out, feature

class Refine(nn.Module):
    def __init__(self):
        super(Refine, self).__init__()
        self.v0 = nn.Sequential(
                nn.Conv2d(64, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 4, 3, padding=1),
                nn.ReLU(inplace=True),
            )
        self.v1 = nn.Sequential(
                nn.Conv2d(256, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 16, 3, padding=1),
                nn.ReLU(inplace=True),
            )
        self.v2 = nn.Sequential(
                nn.Conv2d(512, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 32, 3, padding=1),
                nn.ReLU(inplace=True),
            )
        self.h2 = nn.Sequential(
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(inplace=True),
            )
        self.h1 = nn.Sequential(
                nn.Conv2d(16, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, 3, padding=1),
                nn.ReLU(inplace=True),
            )
        self.h0 = nn.Sequential(
                nn.Conv2d(4, 4, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(4, 4, 3, padding=1),
                nn.ReLU(inplace=True),
            )
        
        self.deconv = nn.ConvTranspose2d(256, 32, 15, 15)
        self.post0 = nn.Conv2d(32, 16, 3, padding=1)
        self.post1 = nn.Conv2d(16, 4, 3, padding=1)
        self.post2 = nn.Conv2d(4, 1, 3, padding=1)

    def forward(self, f, corr_feature, pos):
        p0 = F.pad(f[0], [16, 16, 16, 16])[:, :, 4*pos[0]:4*pos[0]+61, 4*pos[1]:4*pos[1]+61]
        p1 = F.pad(f[1], [8, 8, 8, 8])[:, :, 2*pos[0]:2*pos[0]+31, 2*pos[1]:2*pos[1]+31]
        p2 = F.pad(f[2], [4, 4, 4, 4])[:, :, pos[0]:pos[0]+15, pos[1]:pos[1]+15]

        p3 = corr_feature[:, :, pos[0], pos[1]].view(-1, 256, 1, 1)

        out = self.deconv(p3)
        out = self.post0(F.upsample(self.h2(out) + self.v2(p2), size=(31, 31)))
        out = self.post1(F.upsample(self.h1(out) + self.v1(p1), size=(61, 61)))
        out = self.post2(F.upsample(self.h0(out) + self.v0(p0), size=(127, 127)))
        out = out.view(-1, 127*127)
        return out
