# -*- coding: utf-8 -*-
import copy
import os.path as osp
from typing import Dict, List

import cv2
import numpy as np
from loguru import logger
from yacs.config import CfgNode

from siamfcpp.data.dataset.dataset_base import TRACK_DATASETS, DatasetBase
from siamfcpp.evaluation.got_benchmark.datasets import GOT10k
from siamfcpp.pipeline.utils.bbox import xywh2xyxy

_current_dir = osp.dirname(osp.realpath(__file__))


@TRACK_DATASETS.register
class GOT10kDataset(DatasetBase):
    r"""
    GOT-10k dataset helper

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
        dataset_root="datasets/GOT-10k",
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
        super(GOT10kDataset, self).__init__()
        self._state["dataset"] = None

    def update_params(self):
        r"""
        an interface for update params
        """
        dataset_root = osp.realpath(self._hyper_params["dataset_root"])
        subset = self._hyper_params["subset"]
        check_integrity = self._hyper_params["check_integrity"]
        self._state["dataset"] = GOT10k(dataset_root,
                                        subset=subset,
                                        check_integrity=check_integrity)
        #print('Done-/siamfc/data/dataset/dataset_impl/got10k.py')

    def __getitem__(self, item: int) -> Dict:
        img_files, anno = self._state["dataset"][item]

        anno = xywh2xyxy(anno)
        sequence_data = dict(image=img_files, anno=anno)

        return sequence_data

    def __len__(self):
        return len(self._state["dataset"])


@TRACK_DATASETS.register
class GOT10kDatasetFixed(GOT10kDataset):
    r"""Inherited from GOT10kDataset with exclusion of unfixed sequence
    When sampled sequence is within unfixed list, it will resample another dataset 
        until the sampled sequence is not a unfixed sequnece.
    """
    extra_hyper_params = dict(
        unfixed_list=osp.join(_current_dir, "utils/unfixed_got10k_list.txt"))

    def __init__(self) -> None:
        super(GOT10kDatasetFixed, self).__init__()

    def update_params(self):
        r"""
        an interface for update params
        """
        super(GOT10kDatasetFixed, self).update_params()
        unfixed_list_file = self._hyper_params["unfixed_list"]
        self._state["unfixed_list"] = self._read_unfixed_list(unfixed_list_file)

    def __getitem__(self, item: int) -> Dict:
        sequence_data = super(GOT10kDatasetFixed, self).__getitem__(item)
        while self._is_unfixed_sequence(sequence_data):
            item = self._resample_item(item)
            sequence_data = super(GOT10kDatasetFixed, self).__getitem__(item)

        return sequence_data

    def _read_unfixed_list(self, file: str) -> List[str]:
        """read unfixed list of GOT-10k
        
        Parameters
        ----------
        file : str
            unfixed list file
        
        Returns
        -------
        List[str]
            list of video name
        """
        with open(file, "r") as f:
            l = f.readlines()
        l = [s.strip() for s in l]

        return l

    def _is_unfixed_sequence(self, sequence_data):
        img_file = sequence_data["image"][0]
        seq_dir = osp.dirname(img_file)
        seq_name = osp.basename(seq_dir)
        is_unfixed = (seq_name in self._state["unfixed_list"])
        if is_unfixed:
            logger.info("Unfixed GOT10k sequence sampled at: %s" % seq_dir)

        return is_unfixed

    def _resample_item(self, item: int):
        if "rng" not in self._state:
            self._state["rng"] = np.random.RandomState(item)
        rng = self._state["rng"]
        new_item = rng.choice(len(self))

        return new_item


GOT10kDatasetFixed.default_hyper_params = copy.deepcopy(
    GOT10kDatasetFixed.default_hyper_params)
GOT10kDatasetFixed.default_hyper_params.update(
    GOT10kDatasetFixed.extra_hyper_params)
