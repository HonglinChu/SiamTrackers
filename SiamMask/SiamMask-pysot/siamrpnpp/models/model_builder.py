# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch  
import torch.nn as nn
import torch.nn.functional as F

from siamrpnpp.core.config import cfg
from siamrpnpp.models.loss import select_cross_entropy_loss, weight_l1_loss, select_mask_logistic_loss
from siamrpnpp.models.backbone import get_backbone
from siamrpnpp.models.head import get_rpn_head, get_mask_head, get_refine_head
from siamrpnpp.models.neck import get_neck

class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)
        
        ## build adjust layer 
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)
        
        #---------------------------------------------------------(1)---------------------------------------------------#
        
        # build rpn head 
        self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
                                     **cfg.RPN.KWARGS)

        # build mask head
        if cfg.MASK.MASK:
            self.mask_head = get_mask_head(cfg.MASK.TYPE,
                                           **cfg.MASK.KWARGS)
            if cfg.REFINE.REFINE:
                self.refine_head = get_refine_head(cfg.REFINE.TYPE)

    def template(self, z):
        zf = self.backbone(z) #[0, 1, 2, 3]
        if cfg.MASK.MASK:
            zf = zf[-1]  # zf = zf 
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x) #[1, 64, 125, 125], [1, 256, 63, 63], [1, 512, 31, 31], [1, 1024, 31, 31]
        if cfg.MASK.MASK:
            self.xf = xf[:-1] # refine
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf) #[1, 1024, 31, 31] --> [1, 256, 31, 31]
        cls, loc = self.rpn_head(self.zf, xf) #

        if cfg.MASK.MASK:
            if cfg.REFINE.REFINE:
                mask_corr_feature = self.mask_head.forward_corrs(self.zf, xf)
                mask = self.refine_head(f=self.xf_refine, corr_feature=mask_corr_feature)
            else:
                mask = self.mask_head(self.zf, xf)
        
        # if cfg.MASK.MASK:
        #     #mask, self.mask_corr_feature = self.mask_head(self.zf, xf) #
        #     mask = self.mask_head(self.zf, xf) #

        return {
                'cls': cls,
                'loc': loc,
                'mask': mask if cfg.MASK.MASK else None
               }

    def mask_refine(self, pos):
        return self.refine_head(self.xf, self.mask_corr_feature, pos)

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        label_loc_weight = data['label_loc_weight'].cuda()

        # Mask 
        label_mask = data['label_mask'].cuda()
        label_mask_weight = data['label_mask_weight'].cuda()

        # get feature 
        zf = self.backbone(template) 
        xf = self.backbone(search) 

        if cfg.MASK.MASK:
            zf = zf[-1]
            self.xf_refine = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)
        cls, loc = self.rpn_head(zf, xf)

        # get loss
        cls = self.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls, label_cls)
        loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight) 

        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss

        if cfg.MASK.MASK:
            # TODO
            pad=32 
            mask = self.mask_head(zf, xf)

            mask_loss = None
            mask_loss, iou_m, iou_5, iou_7 = select_mask_logistic_loss(mask, label_mask, label_mask_weight, padding=pad)

            outputs['total_loss'] += cfg.TRAIN.MASK_WEIGHT * mask_loss
            outputs['mask_loss'] = mask_loss 
            outputs['mask_iou_mean'] = iou_m
            outputs['mask_iou_5'] = iou_5
            outputs['mask_iou_7'] = iou_7

        # if cfg.MASK.MASK:
        #     padding = 32 # 
        #     if cfg.REFINE.REFINE:
        #         mask_corr_feature = self.mask_head.mask.forward_corrs(zf, xf)
        #         mask = self.refine_head(self.xf_refine,mask_corr_feature)  
        #     else: 
        #         mask= self.mask_head(zf, xf)
        #     mask_loss, iou_m, iou_5, iou_7 = select_mask_logistic_loss(mask, label_mask, label_mask_weight, padding=padding)
        #     outputs['total_loss'] += cfg.TRAIN.MASK_WEIGHT * mask_loss 
        #     outputs['mask_loss'] = mask_loss
        #     outputs['mask_iou_mean'] = iou_m
        #     outputs['mask_iou_at_5'] = iou_5
        #     outputs['mask_iou_at_7'] = iou_7 

        return outputs
