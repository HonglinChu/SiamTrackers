# -*- coding: utf-8 -*
from loguru import logger

import torch

from siamfcpp.model.module_base import ModuleBase
from siamfcpp.model.task_model.taskmodel_base import (TRACK_TASKMODELS,
                                                          VOS_TASKMODELS)

torch.set_printoptions(precision=8)


@VOS_TASKMODELS.register
class SatVOS(ModuleBase):
    r"""
    State-Aware Tracker model for VOS

    Hyper-Parameters
    ----------------
    pretrain_model_path: string
        path to parameter to be loaded into module
    """

    default_hyper_params = dict(pretrain_model_path="", )

    def __init__(self, GML_extractor, joint_encoder, decoder, loss):
        super(SatVOS, self).__init__()
        self.GML_extractor = GML_extractor
        self.joint_encoder = joint_encoder
        self.decoder = decoder
        self.loss = loss

    def forward(self, *args, phase="train"):
        r"""
        Perform VOS process for different phases (e.g. train / global_feature / segment)

        Arguments
        ---------
        filterd_image: torch.Tensor
            filtered image patch for global modeling loop

        saliency_image: torch.Tensor
            saliency image for saliency encoder
        corr_feature: torch.Tensor
            correlated feature produced by siamese encoder
        global_feature: torch.Tensor
            global feature produced by global modeling loop

        Returns
        -------
        f_g: torch.Tensor
            global feature extracted from filtered image
        pred_mask: torch.Tensor
            predicted mask after sigmoid for the patch of saliency image

        """
        # phase: train
        if phase == 'train':
            saliency_image, corr_feature, filtered_image = args
            global_feature = self.GML_extractor(filtered_image)
            enc_features = self.joint_encoder(saliency_image, corr_feature)
            decoder_features = [global_feature] + enc_features
            out_list = self.decoder(decoder_features, phase="train")

        # phase: feature
        elif phase == 'global_feature':
            filtered_image, = args
            f_g = self.GML_extractor(filtered_image)
            out_list = [f_g]

        elif phase == 'segment':
            saliency_image, corr_feature, global_feature = args
            enc_features = self.joint_encoder(saliency_image, corr_feature)
            decoder_features = [global_feature] + enc_features

            outputs = self.decoder(decoder_features, phase="test")
            pred_mask = outputs
            out_list = [pred_mask]

        else:
            raise ValueError("Phase non-implemented.")
        return out_list

    def set_device(self, dev):
        if not isinstance(dev, torch.device):
            dev = torch.device(dev)
        self.to(dev)
        if self.loss is not None:
            for loss_name in self.loss:
                self.loss[loss_name].to(dev)
