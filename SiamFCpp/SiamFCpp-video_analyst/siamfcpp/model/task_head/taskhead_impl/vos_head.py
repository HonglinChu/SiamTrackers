# -*- coding: utf-8 -*

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from siamfcpp.model.common_opr.common_block import projector, upsample_block
from siamfcpp.model.module_base import ModuleBase
from siamfcpp.model.task_head.taskhead_base import VOS_HEADS

torch.set_printoptions(precision=8)


@VOS_HEADS.register
class DecoderHead(ModuleBase):
    r"""
    DecoderHead for SAT

    Hyper-parameter
    ---------------
    output_size: int
        output size of predicted mask

    """

    default_hyper_params = dict(output_size=257,
                                input_channel_list=[512, 256, 128, 64])

    def __init__(self):
        super(DecoderHead, self).__init__()
        self.output_size = self._hyper_params["output_size"]
        self.out_projector = projector(128, 1)
        self.f_s16_projector = projector(256, 1)
        self.f_s8_projector = projector(256, 1)
        self.activation = nn.Sigmoid()

    def update_params(self):
        input_channel_list = self._hyper_params["input_channel_list"]
        self.upblock1 = upsample_block(input_channel_list[0],
                                       input_channel_list[0], 256)
        self.upblock2 = upsample_block(256, input_channel_list[1], 256)
        self.upblock3 = upsample_block(256, input_channel_list[2], 256)
        self.upblock4 = upsample_block(256, input_channel_list[3], 128)

    def forward(self, feature_list, phase="train"):
        x1, x2, x3, x4, x5 = feature_list
        f_s32 = self.upblock1(x1, x2)
        f_s16 = self.upblock2(f_s32, x3)
        f_s8 = self.upblock3(f_s16, x4)
        f_s4 = self.upblock4(f_s8, x5)

        p = self.out_projector(f_s4)
        p_resize = F.interpolate(p, (self.output_size, self.output_size),
                                 mode='bilinear',
                                 align_corners=False)
        if phase == "train":
            pred_s16 = self.f_s16_projector(f_s16)
            pred_s16_resize = F.interpolate(
                pred_s16, (self.output_size, self.output_size),
                mode='bilinear',
                align_corners=False)
            pred_s8 = self.f_s8_projector(f_s8)
            pred_s8_resize = F.interpolate(pred_s8,
                                           (self.output_size, self.output_size),
                                           mode='bilinear',
                                           align_corners=False)
            return [pred_s16_resize, pred_s8_resize, p_resize]
        else:
            prediction = self.activation(p_resize)
            return prediction
