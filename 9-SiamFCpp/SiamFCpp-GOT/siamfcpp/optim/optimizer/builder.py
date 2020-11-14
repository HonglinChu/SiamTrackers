# -*- coding: utf-8 -*-

from yacs.config import CfgNode

import torch
from torch import nn

from siamfcpp.utils import merge_cfg_into_hps

from .optimizer_base import OPTIMIZERS, OptimizerBase


def build(task: str, cfg: CfgNode, model: nn.Module) -> OptimizerBase:
    r"""
    Arguments
    ---------
    task: str
        task name (track|vos)
    cfg: CfgNode
        node name: optim
    """
    name = cfg.name
    module = OPTIMIZERS[name](cfg, model)

    hps = module.get_hps()
    hps = merge_cfg_into_hps(cfg[name], hps)
    module.set_hps(hps)
    module.update_params()

    return module


def get_config() -> CfgNode:
    cfg = CfgNode()
    cfg["name"] = ""
    for name, module in OPTIMIZERS.items():
        cfg[name] = CfgNode()
        hps = module.default_hyper_params
        for hp_name in hps:
            cfg[name][hp_name] = hps[hp_name]

    return cfg
