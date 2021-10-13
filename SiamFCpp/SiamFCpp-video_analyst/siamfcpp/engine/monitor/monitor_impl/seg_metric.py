# -*- coding: utf-8 -*
import itertools
import os
from typing import Dict

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm

import torch

from siamfcpp.utils import dist_utils

from ..monitor_base import VOS_MONITORS, MonitorBase
from .utils import (AverageMeter, fast_hist, labelcolormap, mask_colorize,
                    np_sigmoid)


@VOS_MONITORS.register
class SegMetric(MonitorBase):
    r"""
    metrics for segmentation

    Hyper-parameters
    ----------------
    gt_name: str
        gt name in training data
    img_name: str
        img name in training data
    num_classes: int
        number of classes
    ignore_label: int
        ignore label
    show_items: list
        metric names for show, ["mean_iou", "acc", "acc_cls", "fwavacc"]
    show_predict: bool
        whether show the predict mask
    show_gt: bool
        whether show the gt mask
    max_show_num: int
        the max number of images to show at all
    interval: int
        the interval for claculation
    result_path:
        the path of dir for show image
    """

    default_hyper_params = {
        "gt_name": "seg_mask",
        "img_name": "seg_img",
        "num_classes": 2,
        "ignore_label": 255,
        "show_items": ["mean_iou"],
        "avg_range": 5,
        "show_predict": True,
        "show_gt": True,
        "max_show_num": 10,
        "interval": 1,
        "result_path": ""
    }

    def __init__(self, ):
        r"""
        Arguments
        ---------
        """
        super(SegMetric, self).__init__()
        self.show_id = 0
        self.outputs = {}
        self.metric_dict = {}
        self.color_map = None

    def init(self, engine_state: Dict):
        super(SegMetric, self).init(engine_state)

    def _fast_hist(self, label_pred, label_true, num_classes, ignore_label=255):
        mask = (label_true >= 0) & (label_true < num_classes) & (label_true !=
                                                                 ignore_label)
        hist = np.bincount(
            num_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=num_classes**2).reshape(num_classes, num_classes)
        return hist

    def _draw_predict_mask(self, image, predict, num_classes):
        if self.color_map is None:
            self.color_map = labelcolormap(num_classes)
        mask = mask_colorize(predict, num_classes, self.color_map)
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
        result_image = cv2.addWeighted(image, 0.6, mask, 0.4, 0)
        return result_image

    def update(self, engine_data: Dict):
        r"""
        """
        iter = engine_data["iter"]
        if iter % self._hyper_params["interval"] != 0:
            engine_data["extras"]["seg_metric"] = self.metric_dict
        else:
            # data
            seg_mask = engine_data["training_data"][
                self._hyper_params["gt_name"]]
            images = engine_data["training_data"][
                self._hyper_params["img_name"]]
            predict_data = engine_data["predict_data"][-1]
            extras = engine_data["extras"]
            if "seg_metric" not in extras:
                extras["seg_metric"] = {}
            num_classes = self._hyper_params["num_classes"]
            hist = np.zeros((num_classes, num_classes))
            ignore_label = self._hyper_params["ignore_label"]
            result_dict = {}
            for i, (image, ro,
                    lt) in enumerate(zip(images, predict_data, seg_mask)):
                image = image.cpu().numpy()
                ro = ro.cpu().detach().numpy()
                lt = lt.cpu().numpy()
                ro = ro.transpose(1, 2, 0)
                if ro.shape[0] != lt.shape[0] or ro.shape[1] != lt.shape[1]:
                    ro = cv2.resize(ro, (lt.shape[1], lt.shape[0]))
                lp = np.zeros_like(ro, dtype=np.uint8)
                lp[ro > 0.5] = 1
                lp = lp.squeeze()
                hist += fast_hist(lp.flatten(), lt.flatten(), num_classes,
                                  ignore_label)
                result_dir = self._hyper_params["result_path"]
                if not result_dir:
                    result_dir = "tmp/"
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)
                if self._hyper_params["show_predict"]:
                    image_show = image.transpose(1, 2, 0).astype(np.uint8)
                    show_image = self._draw_predict_mask(
                        image_show, lp, num_classes)
                    cv2.imwrite(
                        os.path.join(result_dir,
                                     "predict{}.png".format(self.show_id)),
                        show_image)
                if self._hyper_params["show_gt"]:
                    image_show = image.transpose(1, 2, 0).astype(np.uint8)
                    show_image = self._draw_predict_mask(
                        image_show, lt, num_classes)
                    cv2.imwrite(
                        os.path.join(result_dir,
                                     "gt{}.png".format(self.show_id)),
                        show_image)
                self.show_id += 1
                if self.show_id >= self._hyper_params["max_show_num"]:
                    self.show_id = 0

            result_dict["acc"] = np.diag(hist).sum() / hist.sum()
            acc_cls = np.diag(hist) / hist.sum(axis=1)
            result_dict["acc_cls"] = np.nanmean(acc_cls)
            iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) -
                                  np.diag(hist))
            result_dict["mean_iou"] = np.nanmean(iu)
            freq = hist.sum(axis=1) / hist.sum()
            result_dict["fwavacc"] = (freq[freq > 0] * iu[freq > 0]).sum()
            for show_key in self._hyper_params["show_items"]:
                if show_key not in self.outputs:
                    self.outputs[show_key] = AverageMeter(
                        self._hyper_params["avg_range"])
                self.outputs[show_key].update(result_dict[show_key])
            for key in self.outputs.keys():
                extras["seg_metric"][key] = self.outputs[key].get_mean()
            self.metric_dict = extras["seg_metric"]
