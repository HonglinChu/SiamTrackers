# -*- coding: utf-8 -*-
import os.path as osp
from typing import Dict

import cv2
import numpy as np
from loguru import logger
from yacs.config import CfgNode

from siamfcpp.data.dataset.dataset_base import TRACK_DATASETS, DatasetBase
from siamfcpp.evaluation.got_benchmark.datasets import TrackingNet
from siamfcpp.pipeline.utils.bbox import xywh2xyxy

_current_dir = osp.dirname(osp.realpath(__file__))


@TRACK_DATASETS.register
class TrackingNetDataset(DatasetBase):
    r"""
    ILSVRC2015-VID dataset helper

    Hyper-parameters
    ----------------
    dataset_root: str
        path to root of the dataset
    subset: str
        dataset split name (train|val|train_val)
    ratio: float
        dataset ratio. used by sampler (data.sampler).
    max_diff: int
        maximum difference in index of a pair of sampled frames 
    check_integrity: bool
        if check integrity of dataset or not
    """
    default_hyper_params = dict(
        dataset_root="datasets/TrackingNet",
        subset="train",
        ratio=1.0,
        max_diff=100,
        check_integrity=True,
    )

    def __init__(self) -> None:
        super(TrackingNetDataset, self).__init__()
        self._state["dataset"] = None

    def update_params(self):
        r"""
        an interface for update params
        """
        dataset_root = osp.realpath(self._hyper_params["dataset_root"])
        subset = self._hyper_params["subset"]
        check_integrity = self._hyper_params["check_integrity"]
        cache_dir = osp.join(dataset_root, "cache/vid")
        self._state["dataset"] = TrackingNet(
            dataset_root,
            subset=subset,
            #  cache_dir=cache_dir,
        )

    def __getitem__(self, item: int) -> Dict:
        img_files, anno = self._state["dataset"][item]
        anno = xywh2xyxy(anno)
        sequence_data = dict(image=img_files, anno=anno)

        return sequence_data

    def __len__(self):
        return len(self._state["dataset"])
