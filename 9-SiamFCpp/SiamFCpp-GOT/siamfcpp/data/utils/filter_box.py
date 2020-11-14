# -*- coding: utf-8 -*-
from typing import Dict

import cv2
import numpy as np
from loguru import logger


def filter_unreasonable_training_boxes(im: np.array, bbox,
                                       config: Dict) -> bool:
    r""" 
    Filter too small,too large objects and objects with extreme ratio
    No input check. Assume that all imput (im, bbox) are valid object

    Arguments
    ---------
    im: np.array
        image, formate=(H, W, C)
    bbox: np.array or indexable object
        bounding box annotation in (x, y, w, h) format
    """
    eps = 1e-6
    im_area = im.shape[0] * im.shape[1]
    _, _, w, h = bbox
    bbox_area = w * h
    bbox_area_rate = bbox_area / im_area
    bbox_ratio = h / (w + eps)

    # valid trainng box condition
    conds = [(config["min_area_rate"] < bbox_area_rate,
              bbox_area_rate < config["max_area_rate"]),
             max(bbox_ratio, 1.0 / max(bbox_ratio, eps)) < config["max_ratio"]]
    # if not all conditions are satisfied, filter the sample
    filter_flag = not all(conds)

    return filter_flag


def filter_unreasonable_training_masks(im: np.array, mask,
                                       config: Dict) -> bool:
    r""" 
    Filter too small,too large objects and objects with extreme ratio
    No input check. Assume that all imput (im, bbox) are valid object

    Arguments
    ---------
    im: np.array
        image, formate=(H, W, C)
    mask: np.array
        mask, formate=(H, W) only have 0 and 1
    """
    eps = 1e-6
    im_area = im.shape[0] * im.shape[1]
    try:
        x, y, w, h = cv2.boundingRect(mask)
    except:
        logger.error("error while loading mask")
        return True

    bbox_area = w * h
    bbox_area_rate = bbox_area / im_area
    bbox_ratio = h / (w + eps)
    # valid trainng box condition
    conds = [(config["min_area_rate"] < bbox_area_rate,
              bbox_area_rate < config["max_area_rate"]),
             max(bbox_ratio, 1.0 / max(bbox_ratio, eps)) < config["max_ratio"]]
    # if not all conditions are satisfied, filter the sample
    filter_flag = not all(conds)

    return filter_flag
