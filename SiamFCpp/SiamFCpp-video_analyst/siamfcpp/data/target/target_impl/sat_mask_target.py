# -*- coding: utf-8 -*-

import cv2
import numpy as np
from yacs.config import CfgNode

from siamfcpp.data.utils.crop_track_pair import crop_track_pair_for_sat
from siamfcpp.pipeline.utils.bbox import xywh2xyxy

from ..target_base import VOS_TARGETS, TargetBase


@VOS_TARGETS.register
class SATMaskTarget(TargetBase):
    """
    target for paper State-Aware Tracker for Real-Time Video Object Segmentation

    Hyper-parameters
    ----------------

    context_amount: float
        the context factor for template image
    max_scale: float
        the max scale change ratio for search image
    max_shift:  float
        the max shift change ratio for search image
    max_scale_temp: float
        the max scale change ratio for template image
    max_shift_temp:  float
        the max shift change ratio for template image
    track_z_size: int
        output size of template image
    track_x_size: int
        output size of search image
    seg_x_size: int
        the original size of segmentation search image
    seg_x_resize: int
        the resized output size of segmentation search image
    global_fea_output: int
        the image size of images for global feature extraction
    """
    default_hyper_params = dict(
        track_z_size=127,
        track_x_size=303,
        seg_x_size=129,
        seg_x_resize=257,
        global_fea_input_size=129,
        context_amount=0.5,
        max_scale=0.3,
        max_shift=0.4,
        max_scale_temp=0.1,
        max_shift_temp=0.1,
    )

    def __init__(self):
        super().__init__()

    def __call__(self, sampled_data):
        data1 = sampled_data["data1"]
        data2 = sampled_data["data2"]
        im_temp, mask_temp = data1["image"], data1["anno"]
        bbox_temp = cv2.boundingRect(mask_temp)
        bbox_temp = xywh2xyxy(bbox_temp)
        im_curr, mask_curr = data2["image"], data2["anno"]
        bbox_curr = cv2.boundingRect(mask_curr)
        bbox_curr = xywh2xyxy(bbox_curr)
        data_dict = crop_track_pair_for_sat(im_temp,
                                            bbox_temp,
                                            im_curr,
                                            bbox_curr,
                                            config=self._hyper_params,
                                            mask_tmp=mask_temp,
                                            mask_curr=mask_curr)
        if sampled_data["is_negative_pair"]:
            data_dict["seg_mask"] *= 0

        return data_dict
