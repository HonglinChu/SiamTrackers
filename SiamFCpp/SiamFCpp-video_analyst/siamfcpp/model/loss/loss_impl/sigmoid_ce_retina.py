# -*- coding: utf-8 -*
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from ...module_base import ModuleBase
from ..loss_base import TRACK_LOSSES
from .utils import SafeLog

eps = np.finfo(np.float32).tiny


@TRACK_LOSSES.register
class SigmoidCrossEntropyRetina(ModuleBase):

    default_hyper_params = dict(
        name="focal_ce",
        background=0,
        ignore_label=-1,
        weight=1.0,
        alpha=0.5,
        gamma=0.0,
    )

    def __init__(self, ):
        super(SigmoidCrossEntropyRetina, self).__init__()
        self.safelog = SafeLog()
        self.register_buffer("t_one", torch.tensor(1., requires_grad=False))

    def update_params(self, ):
        self.background = self._hyper_params["background"]
        self.ignore_label = self._hyper_params["ignore_label"]
        self.weight = self._hyper_params["weight"]
        # focal loss coefficients
        self.register_buffer(
            "alpha",
            torch.tensor(float(self._hyper_params["alpha"]),
                         requires_grad=False))
        self.register_buffer(
            "gamma",
            torch.tensor(float(self._hyper_params["gamma"]),
                         requires_grad=False))

    def forward(self, pred_data, target_data):
        r"""
        Focal loss
        :param pred: shape=(B, HW, C), classification logits (BEFORE Sigmoid)
        :param label: shape=(B, HW)
        """
        r"""
        Focal loss
        Arguments
        ---------
        pred: torch.Tensor
            classification logits (BEFORE Sigmoid)
            format: (B, HW)
        label: torch.Tensor
            training label
            format: (B, HW)

        Returns
        -------
        torch.Tensor
            scalar loss
            format: (,)
        """
        pred = pred_data["cls_pred"]
        label = target_data["cls_gt"]
        mask = ~(label == self.ignore_label)
        mask = mask.type(torch.Tensor).to(label.device)
        vlabel = label * mask
        zero_mat = torch.zeros(pred.shape[0], pred.shape[1], pred.shape[2] + 1)

        one_mat = torch.ones(pred.shape[0], pred.shape[1], pred.shape[2] + 1)
        index_mat = vlabel.type(torch.LongTensor)

        onehot_ = zero_mat.scatter(2, index_mat, one_mat)
        onehot = onehot_[:, :, 1:].type(torch.Tensor).to(pred.device)

        pred = torch.sigmoid(pred)
        pos_part = (1 - pred)**self.gamma * onehot * self.safelog(pred)
        neg_part = pred**self.gamma * (1 - onehot) * self.safelog(1 - pred)
        loss = -(self.alpha * pos_part +
                 (1 - self.alpha) * neg_part).sum(dim=2) * mask.squeeze(2)

        positive_mask = (label > 0).type(torch.Tensor).to(pred.device)

        loss = loss.sum() / torch.max(positive_mask.sum(),
                                      self.t_one) * self._hyper_params["weight"]
        extra = dict()

        return loss, extra


if __name__ == '__main__':
    B = 16
    HW = 17 * 17
    pred_cls = pred_ctr = torch.tensor(
        np.random.rand(B, HW, 1).astype(np.float32))
    pred_reg = torch.tensor(np.random.rand(B, HW, 4).astype(np.float32))

    gt_cls = torch.tensor(np.random.randint(2, size=(B, HW, 1)),
                          dtype=torch.int8)
    gt_ctr = torch.tensor(np.random.rand(B, HW, 1).astype(np.float32))
    gt_reg = torch.tensor(np.random.rand(B, HW, 4).astype(np.float32))

    criterion_cls = SigmoidCrossEntropyRetina()
    loss_cls = criterion_cls(pred_cls, gt_cls)

    criterion_ctr = SigmoidCrossEntropyCenterness()
    loss_ctr = criterion_ctr(pred_ctr, gt_ctr, gt_cls)

    criterion_reg = IOULoss()
    loss_reg = criterion_reg(pred_reg, gt_reg, gt_cls)

    from IPython import embed
    embed()
