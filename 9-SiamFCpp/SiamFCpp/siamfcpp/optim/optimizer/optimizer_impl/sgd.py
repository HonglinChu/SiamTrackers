# -*- coding: utf-8 -*-

import numpy as np
from yacs.config import CfgNode

import torch
from torch import optim

from siamfcpp.data.dataset.dataset_base import DatasetBase
from siamfcpp.evaluation.got_benchmark.datasets import got10k

from ..optimizer_base import OPTIMIZERS, OptimizerBase


@OPTIMIZERS.register
class SGD(OptimizerBase):
    r"""
    Tracking data sampler

    Hyper-parameters
    ----------------
    """
    extra_hyper_params = dict(
        lr=0.1,
        momentum=0.9,
        weight_decay=0.00005,
    )

    def __init__(self, cfg: CfgNode, model: torch.nn.Module) -> None:
        super(SGD, self).__init__(cfg, model)

    def update_params(self, ):
        super(SGD, self).update_params()
        params = self._state["params"]
        kwargs = self._hyper_params
        valid_keys = self.extra_hyper_params.keys()
        kwargs = {k: kwargs[k] for k in valid_keys}
        self._optimizer = optim.SGD(params, **kwargs)


SGD.default_hyper_params.update(SGD.extra_hyper_params)
