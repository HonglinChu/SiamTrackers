from typing import Dict

import cv2
import numpy as np
from loguru import logger
from yacs.config import CfgNode

from siamfcpp.data.utils.filter_box import \
    filter_unreasonable_training_boxes
from siamfcpp.pipeline.utils.bbox import xyxy2xywh

from ..filter_base import TRACK_FILTERS, VOS_FILTERS, FilterBase


@TRACK_FILTERS.register
@VOS_FILTERS.register
class TrackPairFilter(FilterBase):
    r"""
    Tracking data filter

    Hyper-parameters
    ----------------
    """
    default_hyper_params = dict(
        max_area_rate=0.6,
        min_area_rate=0.001,
        max_ratio=10,
        target_type="bbox",
    )

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data: Dict) -> bool:
        if data is None:
            return True
        im, anno = data["image"], data["anno"]
        if self._hyper_params["target_type"] == "bbox":
            bbox = xyxy2xywh(anno)
        elif self._hyper_params["target_type"] == "mask":
            bbox = cv2.boundingRect(anno)
        else:
            logger.error("unspported target type {} in filter".format(
                self._hyper_params["target_type"]))
            exit()
        filter_flag = filter_unreasonable_training_boxes(
            im, bbox, self._hyper_params)
        return filter_flag
