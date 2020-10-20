# -*- coding: utf-8 -*-
from abc import ABCMeta

import cv2 as cv
import numpy as np
from yacs.config import CfgNode

import torch
from torch import nn, optim

from siamfcpp.utils import Registry

GRAD_MODIFIERS = Registry('GRAD_MODIFIER')


class GradModifierBase:
    __metaclass__ = ABCMeta
    r"""
    base class for GradModifier. Reponsible for scheduling optimizer (learning rate) during training

    Define your hyper-parameters here in your sub-class.
    """
    default_hyper_params = dict()

    def __init__(self, ) -> None:
        r"""
        GradModifier, reponsible for scheduling optimizer

        Arguments
        ---------
        cfg: CfgNode
            data config, including cfg for datasset / sampler
        
        s: List[DatasetBase]
            collections of datasets
        seed: int
            seed to initialize random number generator
            important while using multi-worker data loader
        """
        self._hyper_params = self.default_hyper_params
        self._state = dict()
        self._model = None
        self._optimizer = None

    def get_hps(self) -> dict:
        r"""
        Getter function for hyper-parameters

        Returns
        -------
        dict
            hyper-parameters
        """
        return self._hyper_params

    def set_hps(self, hps: dict) -> None:
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
    def modify_grad(self, module: nn.Module, epoch: int, iteration: int = -1):
        r"""
        Schedule the underlying optimizer/model
        
        Parameters
        ----------
        epoch : int
            [description]
        iteration : int
            [description]
        Returns
        -------
        Dict:
            dict containing the schedule state
        """
