from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import os

__all__ = ['SiamFCNet']


class SiamFCNet(nn.Module):

    def __init__(self, backbone, head):
        super(SiamFCNet, self).__init__()
        self.features = backbone
        self.head = head
    
    def forward(self, z, x):
        z = self.features(z) # [8, 512, 5, 5]
        x = self.features(x)
        return self.head(z, x)  

