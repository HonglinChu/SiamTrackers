# -*- coding: utf-8 -*-
import json
import re

import cv2
import numpy as np
from yacs.config import CfgNode

import torch
from torch import nn

from ..grad_modifier_base import GRAD_MODIFIERS, GradModifierBase
from .utils.freeze import apply_freeze_schedule


@GRAD_MODIFIERS.register
class DynamicFreezer(GradModifierBase):
    r"""
    Learning rate scheduler, including:
    - learning rate adjusting
    - learning rate multiplying

    Hyper-parameters
    ----------------
    phases: Dict

    """
    default_hyper_params = dict(schedule=[], )

    def __init__(self, ) -> None:
        super().__init__()

    def update_params(self) -> None:
        r"""
        Resolve dynamic freezing schedule
        """
        cfg = self._hyper_params["schedule"]
        if len(cfg) > 0:
            schedule = list()
            for freeze_str in cfg:
                mult_cfg = json.loads(freeze_str)
                compiled_regex = re.compile(mult_cfg["regex"])
                mult_cfg["compiled_regex"] = compiled_regex
                schedule.append(mult_cfg)
            self._state["schedule"] = schedule

    def modify_grad(self, module: nn.Module, epoch: int, iteration: int = -1):
        if (iteration < 0) and ("schedule" in self._state):
            # epoch-level scheduling
            apply_freeze_schedule(module, epoch, self._state["schedule"])
        else:
            # iteration-level scheduling
            pass
