# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from siamcar.core.config import cfg
from siamcar.models.loss_car import make_siamcar_loss_evaluator
from siamcar.models.backbone import get_backbone
from siamcar.models.head.car_head import CARHead
from siamcar.models.neck import get_neck
from siamcar.utils.location_grid import compute_locations
from siamcar.utils.xcorr import xcorr_depthwise

class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer  AdjustAllLayer 
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build car head
        self.car_head = CARHead(cfg, 256)

        # build response map
        self.xcorr_depthwise = xcorr_depthwise

        # build loss
        self.loss_evaluator = make_siamcar_loss_evaluator(cfg)
        # 逆卷积，这里仅仅是为了通道压缩,
        self.down = nn.ConvTranspose2d(256 * 3, 256, 1, 1)
    
    # （1）初始帧 获取模板分支特征
    def template(self, z):

        # (1) 骨干网络： resnet， Alexnet
        zf = self.backbone(z) # resnet：[[1, 512, 15, 15]， [1,1024, 15, 15]，[1, 2048, 15, 15]]

        # (2) 通道压缩到256，特征图中心裁剪【15， 15】--> [7, 7]  基本可以代表目标区域的大小，包含很少的背景信息
        if cfg.ADJUST.ADJUST: # 特征增强
            zf = self.neck(zf) # [[1, 512, 15, 15]， [1,1024, 15, 15]，[1, 2048, 15, 15]] --> [1, 256, 7, 7]， [1,256, 7, 7]，[1, 256, 7, 7]]

        self.zf = zf

    # （2）搜索帧 获取搜索分支特征
    def track(self, x):
        xf = self.backbone(x) # [[1, 512, 31, 31]， [1,1024, 31, 31]，[1, 2048, 31, 31]]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf) # 通道压缩 [[1, 512, 31, 31]， [1,1024, 31, 31]，[1, 2048, 31, 31]] --> [[1, 256, 31, 31]， [1,256, 31, 31]，[1, 256, 31, 31]]
        
        # 3组特征相关操作，通道拼接 
        if cfg.BACKBONE.TYPE=='resnet50':
            features = self.xcorr_depthwise(xf[0],self.zf[0]) #[1, 256, 31, 31]  [1,256, 7,7] --> [1, 256, 25, 25]
            for i in range(len(xf)-1):
                features_new = self.xcorr_depthwise(xf[i+1],self.zf[i+1])
                features = torch.cat([features,features_new],1)  # [1, 768, 25, 25]
            # 使用1x1卷积进行通道降维，这里为何要用逆卷积
            features = self.down(features) # [1, 768, 25, 25] -->  [1, 256, 25, 25]

        elif cfg.BACKBONE.TYPE=='alexnet':
            features = self.xcorr_depthwise(xf,self.zf) #[1, 256, 31, 31]  [1,256, 7,7] --> [1, 256, 25, 25]
            
        # 重点FCOS
        cls, loc, cen = self.car_head(features) #[1, 256, 25, 25]

        return {
                'cls': cls,
                'loc': loc,
                'cen': cen
               } 

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4) # 把要操作的维度调整到后面
        return cls

    # # # 正常训练的时候有使用如下forward 
    def forward(self, data):   # data: template[B,C,TEMP_SIZE,TEMP_SIZE], search[B,C,SEARCH_SIZE,SEARCH_SIZE], label_cls[B,OUTPUT_SIZE,OUTPUT_SIZE], bbox[B,4]
        """ only used in training
        """
        # （1）正常训练模式   
        if data.size(1)!=3 and data.size(0)!=1:
            template = data['template'].cuda()
            search = data['search'].cuda()
            label_cls = data['label_cls'].cuda()
            label_loc = data['bbox'].cuda()
            # get feature
            zf = self.backbone(template) #[32, 3, 127, 127] --> [B,C,,H,W]
            xf = self.backbone(search)#[32, 3, 255, 255]
            if cfg.ADJUST.ADJUST:
                zf = self.neck(zf) # 通道压缩-->256 , 空间维度中心裁剪[15,15]-->[7,7]
                xf = self.neck(xf) # 通道压缩-->256, 空间维度[31,31](对应255的输入大小)
            # 三组特征分别进行相关操作 
            if cfg.BACKBONE.TYPE=='resnet50': # ResNet,三组特征卷积操作 
                features = self.xcorr_depthwise(xf[0],zf[0])
                for i in range(len(xf)-1):
                    features_new = self.xcorr_depthwise(xf[i+1],zf[i+1])
                    features = torch.cat([features,features_new],1)
                features = self.down(features) # 通道压缩 768 --> 256

            elif cfg.BACKBONE.TYPE=='alexnet': # AlexNet
                features = self.xcorr_depthwise(xf,zf) # [32, 256, 22, 22], [32, 256,6,6]

            cls, loc, cen = self.car_head(features) #[32, 256, 25, 25]--> cls[B,2,25,25], loc[B,4,25,25], cen[B,1,25,25]
            locations = compute_locations(cls, cfg.TRACK.STRIDE) # 这一步涉及特征图到原图的映射，非常重要。STRIDE=8
            cls = self.log_softmax(cls) #[32,2,25,25] --> [32,1,25,25,2]
            cls_loss, loc_loss, cen_loss = self.loss_evaluator(
                locations,
                cls,
                loc,
                cen, label_cls, label_loc
            )
            # get loss
            outputs = {}
            outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
                cfg.TRAIN.LOC_WEIGHT * loc_loss + cfg.TRAIN.CEN_WEIGHT * cen_loss
            outputs['cls_loss'] = cls_loss
            outputs['loc_loss'] = loc_loss
            outputs['cen_loss'] = cen_loss
            return outputs

        # 计算Flops， Flops_Params.py 
        elif data.size(1)==3 and data.size(0)==1:
            xf = self.backbone(data) # [[1, 512, 31, 31]， [1,1024, 31, 31]，[1, 2048, 31, 31]]
            if cfg.ADJUST.ADJUST:
                xf = self.neck(xf) # 通道压缩 [[1, 512, 31, 31]， [1,1024, 31, 31]，[1, 2048, 31, 31]] --> [[1, 256, 31, 31]， [1,256, 31, 31]，[1, 256, 31, 31]]
            
            # 3组特征相关操作，通道拼接 
            if cfg.BACKBONE.TYPE=='resnet50':
                features = self.xcorr_depthwise(xf[0],self.zf[0]) #[1, 256, 31, 31]  [1,256, 7,7] --> [1, 256, 25, 25]
                for i in range(len(xf)-1):
                    features_new = self.xcorr_depthwise(xf[i+1],self.zf[i+1])
                    features = torch.cat([features,features_new],1)  # [1, 768, 25, 25]
                # 使用1x1卷积进行通道降维，这里为何要用逆卷积
                features = self.down(features) # [1, 768, 25, 25] -->  [1, 256, 25, 25]

            elif cfg.BACKBONE.TYPE=='alexnet':
                features = self.xcorr_depthwise(xf,self.zf) #[1, 256, 31, 31]  [1,256, 7,7] --> [1, 256, 25, 25]
                
            # 重点FCOS
            cls, loc, cen = self.car_head(features) #[1, 256, 25, 25]

            return {
                    'cls': cls,
                    'loc': loc,
                    'cen': cen
                } 
