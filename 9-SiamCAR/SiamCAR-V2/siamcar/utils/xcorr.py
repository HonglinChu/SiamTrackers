# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn 
import torch.nn.functional as F

def xcorr_slow(x, kernel):
    """for loop to calculate cross correlation, slow version
    """
    batch = x.size()[0]
    out = []
    for i in range(batch):
        px = x[i]
        pk = kernel[i]
        px = px.view(1, px.size()[0], px.size()[1], px.size()[2])
        pk = pk.view(-1, px.size()[1], pk.size()[1], pk.size()[2])
        po = F.conv2d(px, pk)
        out.append(po)
    out = torch.cat(out, 0)
    return out


def xcorr_fast(x, kernel):
    """group conv2d to calculate cross correlation, fast version
    """
    batch = kernel.size()[0]
    pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
    px = x.view(1, -1, x.size()[2], x.size()[3])
    po = F.conv2d(px, pk, groups=batch)
    po = po.view(batch, -1, po.size()[2], po.size()[3])
    return po


def xcorr_depthwise(x, kernel):  # alexnet: [1, 256,22,22], [1, 256,6,6] --> [1, 8192, 17, 17]      resnet50: [32, 256, 31, 31]   [32,256,7,7]
    """depthwise cross correlation
    """
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch*channel, x.size(2), x.size(3))
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch*channel)
    out = out.view(batch, channel, out.size(2), out.size(3)) #[32, 256, 17, 17]
    return out

class DepthwiseXCorr(nn.Module): 
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DepthwiseXCorr, self).__init__()
        # 输入和输出都是256
        self.conv_kernel = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                )
        self.conv_search = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                )

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel) # [32,256,6,6]  -->[32,256,4,4]
        search = self.conv_search(search) # [32,256,22,22]-->[32, 256,20,20]

        feature = xcorr_depthwise(search, kernel) #[32, 256, 17, 17]
        return feature