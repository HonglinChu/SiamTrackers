# -*- coding: utf-8 -*
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from ...module_base import ModuleBase

eps = np.finfo(np.float32).tiny


class SafeLog(ModuleBase):
    r"""
    Safly perform log operation 
    """
    default_hyper_params = dict()

    def __init__(self):
        super(SafeLog, self).__init__()
        self.register_buffer("t_eps", torch.tensor(eps, requires_grad=False))

    def forward(self, t):
        return torch.log(torch.max(self.t_eps, t))
