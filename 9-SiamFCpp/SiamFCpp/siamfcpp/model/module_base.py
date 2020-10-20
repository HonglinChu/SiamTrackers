# -*- coding: utf-8 -*
from copy import deepcopy

from loguru import logger

import torch
from torch import nn

from siamfcpp.utils import md5sum

from .utils.load_state import (filter_reused_missing_keys,
                               get_missing_parameters_message,
                               get_unexpected_parameters_message)


class ModuleBase(nn.Module):
    r"""
    Module/component base class
    """
    # Define your default hyper-parameters here in your sub-class.
    default_hyper_params = dict(pretrain_model_path="")

    def __init__(self):
        super(ModuleBase, self).__init__()
        self._hyper_params = deepcopy(self.default_hyper_params)

    def get_hps(self) -> dict():
        r"""
        Getter function for hyper-parameters

        Returns
        -------
        dict
            hyper-parameters
        """
        return self._hyper_params

    def set_hps(self, hps: dict()) -> None:
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

    def update_params(self):
        model_file = self._hyper_params.get("pretrain_model_path", "")
        if model_file != "":
            state_dict = torch.load(model_file,
                                    map_location=torch.device("cpu"))
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            self.load_model_param(state_dict)
            logger.info(
                "Load pretrained {} parameters from: {} whose md5sum is {}".
                format(self.__class__.__name__, model_file, md5sum(model_file)))

    def load_model_param(self, checkpoint_state_dict):
        model_state_dict = self.state_dict()
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    logger.warning(
                        "'{}' has shape {} in the checkpoint but {} in the "
                        "model! Skipped.".format(k, shape_checkpoint,
                                                 shape_model))
                    checkpoint_state_dict.pop(k)
        # pyre-ignore
        incompatible = self.load_state_dict(checkpoint_state_dict, strict=False)
        if incompatible.missing_keys:
            missing_keys = filter_reused_missing_keys(self,
                                                      incompatible.missing_keys)
            if missing_keys:
                logger.warning(get_missing_parameters_message(missing_keys))
        if incompatible.unexpected_keys:
            logger.warning(
                get_unexpected_parameters_message(incompatible.unexpected_keys))
