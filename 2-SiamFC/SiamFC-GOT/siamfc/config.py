import os 
import torch

import numpy as np     
class Config:
    out_scale=0.001
    exemplar_sz= 127 #第一帧
    instance_sz=255 #
    context= 0.5
    # inference parameters
    scale_num= 3  #尺度因子3
    scale_step= 1.0375
    scale_lr= 0.59
    scale_penalty= 0.9745
    window_influence= 0.176
    response_sz= 17
    response_up= 16
    total_stride= 8
    # train parameters
    epoch_num= 50
    batch_size= 8   #  单机单卡=8, 单机多卡=16   
    num_workers= 8  #
    initial_lr= 1e-2 #0.01
    ultimate_lr=1e-5#0.000052
    weight_decay= 5e-4#正则参数
    momentum= 0.9
    r_pos= 16  #
    r_neg= 0
    def update(self, cfg):
        for k, v in cfg.items():
            setattr(self, k, v)

config=Config()
