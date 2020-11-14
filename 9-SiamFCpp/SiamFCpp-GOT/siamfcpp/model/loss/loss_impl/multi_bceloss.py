# -*- coding: utf-8 -*
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from ...module_base import ModuleBase
from ..loss_base import VOS_LOSSES
from .utils import SafeLog

eps = np.finfo(np.float32).tiny


@VOS_LOSSES.register
class MultiBCELoss(ModuleBase):

    default_hyper_params = dict(
        name="multi_bceloss",
        sub_loss_weights=[0.1, 0.3, 1.0],
        weight=1.0,
        top_ratio=0.0,
    )

    def __init__(self):
        super(MultiBCELoss, self).__init__()

    def update_params(self, ):
        self.sub_loss_weights = self._hyper_params["sub_loss_weights"]
        self.weight = self._hyper_params["weight"]

    def forward(self, pred_data_list, target_data):
        total_loss = 0
        assert len(pred_data_list) == len(self.sub_loss_weights)
        for pred_data, sub_loss_weight in zip(pred_data_list,
                                              self.sub_loss_weights):
            loss = F.binary_cross_entropy_with_logits(pred_data.flatten(),
                                                      target_data.flatten(),
                                                      reduction="none")
            if self._hyper_params["top_ratio"] > 0 and self._hyper_params[
                    "top_ratio"] < 1.0:
                loss, _ = loss.topk(
                    int(loss.numel() * self._hyper_params["top_ratio"]))
            loss = loss.mean() * sub_loss_weight
            total_loss += loss
        extra = dict()
        return total_loss * self.weight, extra
