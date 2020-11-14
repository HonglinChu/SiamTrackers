# -*- coding: utf-8 -*

import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_bn_relu(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=1,
                 kszie=3,
                 pad=0,
                 has_bn=True,
                 has_relu=True,
                 bias=True,
                 groups=1):
        r"""
        Basic block with one conv, one bn, one relu in series.

        Arguments
        ---------
        in_channel: int
            number of input channels
        out_channel: int
            number of output channels
        stride: int
            stride number
        kszie: int
            kernel size
        pad: int
            padding on each edge
        has_bn: bool
            use bn or not
        has_relu: bool
            use relu or not
        bias: bool
            conv has bias or not
        groups: int or str
            number of groups. To be forwarded to torch.nn.Conv2d
        """
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_channel,
                              out_channel,
                              kernel_size=kszie,
                              stride=stride,
                              padding=pad,
                              bias=bias,
                              groups=groups)

        if has_bn:
            self.bn = nn.BatchNorm2d(out_channel)
        else:
            self.bn = None

        if has_relu:
            self.relu = nn.ReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def xcorr_depthwise(x, kernel):
    r"""
    Depthwise cross correlation. e.g. used for template matching in Siamese tracking network

    Arguments
    ---------
    x: torch.Tensor
        feature_x (e.g. search region feature in SOT)
    kernel: torch.Tensor
        feature_z (e.g. template feature in SOT)

    Returns
    -------
    torch.Tensor
        cross-correlation result
    """
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch * channel, x.size(2), x.size(3))
    kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch * channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out


class upsample_block(nn.Module):
    r"""
    Upsample block. e.g. used for upsample and feature fusion in decoder
    """
    def __init__(self, h_channel, l_channel, out_channel):
        r"""
        h_channel:
            channel number of high-level feature

        l_channel:
            channel number of low-level feature

        out_channel:
            channel number of output feature after fusion
        """

        super(upsample_block, self).__init__()
        self.conv1 = conv_bn_relu(h_channel, out_channel, pad=1, bias=False)
        self.conv_adjust = conv_bn_relu(out_channel + l_channel,
                                        out_channel,
                                        pad=1,
                                        bias=False)

    def forward(self, high_level_f, low_level_f):
        r"""
        :param high_level_f: torch.Tensor
            high level feature with smaller resolution

        :param low_level_f: torch.Tensor
            low level feature with larger resolution

        Returns
        -------
        torch.Tensor
            feature fusion result
        """
        high_level_f = self.conv1(high_level_f)
        f_resize = F.interpolate(high_level_f,
                                 size=low_level_f.size()[2:],
                                 mode='bilinear',
                                 align_corners=False)
        f_fusion = torch.cat([f_resize, low_level_f], 1)
        f_adjust = self.conv_adjust(f_fusion)
        return f_adjust


class projector(nn.Module):
    r"""
    Projection layer to adjust channel number
    """
    def __init__(self, in_channel, out_channel):
        super(projector, self).__init__()
        self.conv1 = conv_bn_relu(in_channel,
                                  out_channel,
                                  pad=1,
                                  has_relu=False,
                                  bias=False)

    def forward(self, x):
        x = self.conv1(x)
        return x
