import torch
import os
import torch.nn as nn 
class SiamRPNNet(nn.Module):
    def __init__(self, backbone,head):
        super(SiamRPNNet, self).__init__()

        self.features = backbone

        self.head = head 

    def forward(self, z, x):
        
       # N = template.size(0) # N=32
        z = self.features(z)#[32,256,6,6] 
        x = self.features(x)#[32,256,24,24]
        
        return  self.head(z,x)

    def track_init(self, z):
        #N = template.size(0)
        z = self.features(z)# 输出 [1, 256, 6, 6]
        self.head.z_branch(z)

    def track(self, x):
        #N = detection.size(0)
        x = self.features(x)# 1,256,22,22
                            
        return self.head.x_branch(x)
