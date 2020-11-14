# -*- coding: utf-8 -*-

from abc import ABCMeta
from typing import Dict

import cv2 as cv
import numpy as np
from yacs.config import CfgNode

import torch

from siamfcpp.utils import Registry

TRACK_TARGETS = Registry('TRACK_TARGETS')
VOS_TARGETS = Registry('VOS_TARGETS')

TASK_TARGETS = dict(
    track=TRACK_TARGETS,
    vos=VOS_TARGETS,
)


class TargetBase:
    __metaclass__ = ABCMeta
    r"""
    Target maker. 
    Responsible for transform image (e.g. HWC -> 1CHW), generating training target, etc.

    Define your hyper-parameters here in your sub-class.
    """
    default_hyper_params = dict()

    def __init__(self) -> None:
        r"""
        Target, reponsible for generate training target tensor

        Arguments
        ---------
        cfg: CfgNode
            node name target
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

    def update_params(self) -> None:
        r"""
        an interface for update params
        """
    def __call__(self, sampled_data: Dict) -> Dict:
        r"""
        An interface to mkae target

        Arguments
        ---------
        training_data: Dict
            data whose training target will be made
        """
        for k in sampled_data:
            sampled_data[k] = torch.from_numpy(np.array(sampled_data[k]))

        return sampled_data
