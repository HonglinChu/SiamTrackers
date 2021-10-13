# -*- coding: utf-8 -*-

from abc import ABCMeta
from typing import Dict, List

import cv2 as cv
import numpy as np
from loguru import logger
from yacs.config import CfgNode

from siamfcpp.utils import Registry

from ..dataset.dataset_base import DatasetBase

TRACK_SAMPLERS = Registry('TRACK_SAMPLERS')
VOS_SAMPLERS = Registry('VOS_SAMPLERS')

TASK_SAMPLERS = dict(
    track=TRACK_SAMPLERS,
    vos=VOS_SAMPLERS,
)


class SamplerBase:
    __metaclass__ = ABCMeta
    r"""
    base class for Sampler. Reponsible for sampling from multiple datasets and forming training pair / sequence.

    Define your hyper-parameters here in your sub-class.
    """
    default_hyper_params = dict()

    def __init__(self, datasets: List[DatasetBase] = [], seed: int = 0) -> None:
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
        self._state["rng"] = np.random.RandomState(seed)
        self.datasets = datasets
        for d in datasets:
            dataset_name = type(d).__name__
            logger.info("Sampler's underlying datasets: {}, length {}".format(
                dataset_name, len(d)))

    def get_hps(self) -> Dict:
        r"""
        Getter function for hyper-parameters

        Returns
        -------
        dict
            hyper-parameters
        """
        return self._hyper_params

    def set_hps(self, hps: Dict) -> None:
        r"""
        Set hyper-parameters

        Arguments
        ---------
        hps: dict
            dict of hyper-parameters, the keys must in self.__hyper_params__
        """
        for key in hps:
            if key not in self._hyper_params:
                raise KeyError
            self._hyper_params[key] = hps[key]

    def update_params(self) -> None:
        r"""
        an interface for update params
        """
    def __getitem__(self, item) -> Dict:
        r"""
        An interface to sample data
        """
