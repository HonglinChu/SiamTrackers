# -*- coding: utf-8 -*-

from abc import ABCMeta
from typing import Dict

import cv2 as cv
import numpy as np
from yacs.config import CfgNode

from siamfcpp.data.dataset.dataset_base import DatasetBase
from siamfcpp.utils import Registry

TRACK_FILTERS = Registry('TRACK_FILTERS')
VOS_FILTERS = Registry('VOS_FILTERS')

TASK_FILTERS = dict(
    track=TRACK_FILTERS,
    vos=VOS_FILTERS,
)


class FilterBase:
    __metaclass__ = ABCMeta
    r"""
    base class for Filter. Reponsible for filtering invalid sampled data (e.g. samples with extreme size / ratio)

    Define your hyper-parameters here in your sub-class.
    """
    default_hyper_params = dict()

    def __init__(self) -> None:
        r"""
        Dataset Sampler, reponsible for sampling from different dataset

        Arguments
        ---------
        cfg: CfgNode
            data config, including cfg for datasset / sampler
        datasets: List[DatasetBase]
            collections of datasets
        seed: int
            seed to initialize random number generator
            important while using multi-worker data loader
        """
        self._hyper_params = self.default_hyper_params
        self._state = dict()

    def get_hps(self) -> Dict:
        r"""
        Getter function for hyper-parameters

        Returns
        -------
        Dict
            hyper-parameters
        """
        return self._hyper_params

    def set_hps(self, hps: Dict) -> None:
        r"""
        Set hyper-parameters

        Arguments
        ---------
        hps: Dict
            Dict of hyper-parameters, the keys must in self.__hyper_params__
        """
        for key in hps:
            if key not in self._hyper_params:
                raise KeyError
            self._hyper_params[key] = hps[key]

    def update_params(self):
        r"""
        an interface for update params
        """
    def __call__(self, data: Dict) -> bool:
        r"""
        An interface to filter data

        Arguments
        ---------
        data: Dict
            data to be filter
        
        Returns
        -------
        bool
            True if data should be filtered
            False if data is valid
        """
