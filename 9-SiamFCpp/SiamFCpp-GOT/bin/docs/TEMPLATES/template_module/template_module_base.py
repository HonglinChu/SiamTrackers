# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
from typing import List, Dict

import cv2 as cv
import numpy as np

from yacs.config import CfgNode

from videoanalyst.utils import Registry

TRACK_TEMPLATE_MODULES = Registry('TRACK_TEMPLATE_MODULE')
VOS_TEMPLATE_MODULES = Registry('VOS_TEMPLATE_MODULE')

TASK_TEMPLATE_MODULES = dict(
    track=TRACK_TEMPLATE_MODULES,
    vos=VOS_TEMPLATE_MODULES,
)


class TemplateModuleBase:
    __metaclass__ = ABCMeta
    r"""
    base class for TemplateModule. Reponsible for building tempalte module

    Define your hyper-parameters here in your sub-class.
    """
    default_hyper_params = dict()

    def __init__(self, ) -> None:
        r"""
        Template Module Base Class
        """
        self._hyper_params = self.default_hyper_params
        self._state = dict()

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
