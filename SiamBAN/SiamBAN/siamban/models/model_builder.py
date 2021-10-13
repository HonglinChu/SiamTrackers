# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch 
import torch.nn as nn
import torch.nn.functional as F

from siamban.core.config import cfg
from siamban.models.loss import select_cross_entropy_loss, select_iou_loss
from siamban.models.backbone import get_backbone
from siamban.models.head import get_ban_head
from siamban.models.neck import get_neck


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build ban head
        if cfg.BAN.BAN:
            self.ban_head = get_ban_head(cfg.BAN.TYPE,
                                     **cfg.BAN.KWARGS)

        self.centerness_loss_func = nn.BCEWithLogitsLoss()
    
    def compute_centerness_targets(self, reg_targets): #[xxx, 4]
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]

        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def select_cen_loss(self,cen, label_loc, label_cls):
        
        label_cls = label_cls.reshape(-1)
        pos = label_cls.data.eq(1).nonzero().squeeze().cuda()

        # pred_loc = pred_loc.permute(0, 2, 3, 1).reshape(-1, 4)
        # pred_loc = torch.index_select(pred_loc, 0, pos)
        cen_preds = cen.view(-1)[pos]

        label_loc = label_loc.permute(0, 2, 3, 1).reshape(-1, 4)
        label_loc = torch.index_select(label_loc, 0, pos)
       
        cen_targets=self.compute_centerness_targets(label_loc)
       
        return self.centerness_loss_func(cen_preds,cen_targets)


    def template(self, z):
        zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        cls, loc = self.ban_head(self.zf, xf)
        return {
                'cls': cls,
                'loc': loc
               }
       
        return {
                'cls': cls,
                'loc': loc,
               }

    def log_softmax(self, cls):
        if cfg.BAN.BAN:
            cls = cls.permute(0, 2, 3, 1).contiguous()
            cls = F.log_softmax(cls, dim=3)
        return cls

    def forward(self, data):
        """ only used in training
        """
        if len(data)>=4:
            template = data['template'].cuda()
            search = data['search'].cuda()
            label_cls = data['label_cls'].cuda()
            label_loc = data['label_loc'].cuda()

            # get feature 
            zf = self.backbone(template) #[64,3,127, 127] --> [64, 96, 8.,8]
            xf = self.backbone(search)    # [64,3, 255,255] --> [64, 96, 16,16]
            if cfg.ADJUST.ADJUST:
                zf = self.neck(zf)   #[64, 96, 4, 4]
                xf = self.neck(xf)  # 64,96,16,16
            cls, loc = self.ban_head(zf, xf)

            # cls loss with cross entropy loss 
            cls = self.log_softmax(cls)

            cls_loss = select_cross_entropy_loss(cls, label_cls) # [64, 13, 13, 2]  [64,13,13]

            # loc loss with iou loss
            loc_loss = select_iou_loss(loc, label_loc, label_cls) # [64,4,13,13],[64,4,13,13],[64,13,13]

            outputs = {} 

            outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + cfg.TRAIN.LOC_WEIGHT * loc_loss 
            outputs['cls_loss'] = cls_loss
            outputs['loc_loss'] = loc_loss

            return outputs

        else:
                       # get feature
            xf = self.backbone(data)    # [64,3, 255,255] --> [64, 96, 16,16]
            if cfg.ADJUST.ADJUST:
                xf = self.neck(xf)  # 64,96,16,16
            cls, loc = self.ban_head(self.zf, xf)

            return {
                    'cls': cls,
                    'loc': loc,
                }

