# -*- coding: utf-8 -*-
from loguru import logger
from typing import Dict, List, Tuple

import cv2
import numpy as np
from yacs.config import CfgNode

from ..template_module_base import (TRACK_TEMPLATE_MODULES,
                                    VOS_TEMPLATE_MODULES, TemplateModuleBase)


@TRACK_TEMPLATE_MODULES.register
@VOS_TEMPLATE_MODULES.register
class TemplateModuleImplementation(TemplateModuleBase):
    r"""
    Template Module Implementation

    Hyper-parameters
    ----------------
    """
    default_hyper_params = dict(template_module_hp="", )

    def __init__(self, ) -> None:
        super().__init__()

    def update_params(self) -> None:
        pass
