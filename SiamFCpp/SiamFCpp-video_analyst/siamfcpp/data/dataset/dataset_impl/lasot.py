import os.path as osp
from typing import Dict

import cv2
import numpy as np
from yacs.config import CfgNode

from siamfcpp.data.dataset.dataset_base import TRACK_DATASETS, DatasetBase
from siamfcpp.evaluation.got_benchmark.datasets import LaSOT
from siamfcpp.pipeline.utils.bbox import xywh2xyxy


@TRACK_DATASETS.register
class LaSOTDataset(DatasetBase):
    r"""
    LaSOT dataset helper

    Hyper-parameters
    ----------------
    dataset_root: str
        path to root of the dataset
    subset: str
        dataset split name (train|val|test)
    ratio: float
        dataset ratio. used by sampler (data.sampler).
    max_diff: int
        maximum difference in index of a pair of sampled frames 
    check_integrity: bool
        if check integrity of dataset or not
    """
    default_hyper_params = dict(
        dataset_root="datasets/LaSOT",
        subset="train",
        ratio=1.0,
        max_diff=100,
        check_integrity=True,
    )

    def __init__(self) -> None:
        r"""
        Create dataset with config

        Arguments
        ---------
        cfg: CfgNode
            dataset config
        """
        super().__init__()
        self._state["dataset"] = None

    def update_params(self):
        r"""
        an interface for update params
        """
        dataset_root = osp.realpath(self._hyper_params["dataset_root"])
        subset = self._hyper_params["subset"]
        check_integrity = self._hyper_params["check_integrity"]
        self._state["dataset"] = LaSOT(dataset_root,
                                       subset=subset,
                                       check_integrity=check_integrity)

    def __getitem__(self, item: int) -> Dict:
        img_files, anno = self._state["dataset"][item]

        anno = xywh2xyxy(anno)
        sequence_data = dict(image=img_files, anno=anno)

        return sequence_data

    def __len__(self):
        return len(self._state["dataset"])
