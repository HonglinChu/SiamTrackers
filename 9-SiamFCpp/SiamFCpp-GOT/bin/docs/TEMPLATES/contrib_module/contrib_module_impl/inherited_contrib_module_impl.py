# -*- coding: utf-8 -*-
from copy import deepcopy
from typing import Dict, List, Tuple
from loguru import logger
import numpy as np
import cv2

from yacs.config import CfgNode

from ..contrib_module_base import TRACK_CONTRIB_MODULES, VOS_CONTRIB_MODULES, ContribModuleBase


@TRACK_CONTRIB_MODULES.register
@VOS_CONTRIB_MODULES.register
class InheritedContribModuleImplementation(ContribModuleBase):
    r"""
    Contrib Module Implementation

    Hyper-parameters
    ----------------
    """
    extra_hyper_params = dict(inherited_contrib_module_hp="", )

    def __init__(self, ) -> None:
        super().__init__()

    def update_params(self) -> None:
        pass


InheritedContribModuleImplementation.default_hyper_params = deepcopy(
    InheritedContribModuleImplementation.default_hyper_params)
InheritedContribModuleImplementation.default_hyper_params.update(
    InheritedContribModuleImplementation.extra_hyper_params)
