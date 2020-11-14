# -*- coding: utf-8 -*-

from yacs.config import CfgNode

from siamfcpp.utils import merge_cfg_into_hps

from .grad_modifier_base import GRAD_MODIFIERS, GradModifierBase


def build(task: str, cfg: CfgNode) -> GradModifierBase:
    r"""
    Arguments
    ---------
    task: str
        task name (track|vos)
    cfg: CfgNode
        node name: scheduler
    seed: int
        seed for rng initialization
    """

    name = cfg.name
    module = GRAD_MODIFIERS[name]()

    hps = module.get_hps()
    hps = merge_cfg_into_hps(cfg[name], hps)
    module.set_hps(hps)
    module.update_params()

    return module


def get_config() -> CfgNode:
    cfg = CfgNode()
    cfg["name"] = ""

    for name, module in GRAD_MODIFIERS.items():
        cfg[name] = CfgNode()
        hps = module.default_hyper_params
        for hp_name in hps:
            cfg[name][hp_name] = hps[hp_name]
    return cfg
