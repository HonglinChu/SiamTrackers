# -*- coding: utf-8 -*-

from typing import Dict, List

from yacs.config import CfgNode

from siamfcpp.utils import merge_cfg_into_hps

from .target_base import TASK_TARGETS, TargetBase


def build(task: str, cfg: CfgNode) -> TargetBase:
    r"""
    Arguments
    ---------
    task: str
        task
    cfg: CfgNode
        node name: target
    """
    assert task in TASK_TARGETS, "invalid task name"
    MODULES = TASK_TARGETS[task]

    name = cfg.name
    module = MODULES[name]()
    hps = module.get_hps()
    hps = merge_cfg_into_hps(cfg[name], hps)
    module.set_hps(hps)
    module.update_params()

    return module


def get_config(task_list: List) -> Dict[str, CfgNode]:
    cfg_dict = {name: CfgNode() for name in task_list}

    for cfg_name, modules in TASK_TARGETS.items():
        cfg = cfg_dict[cfg_name]
        cfg["name"] = "IdentityTarget"

        for name in modules:
            cfg[name] = CfgNode()
            module = modules[name]
            hps = module.default_hyper_params
            for hp_name in hps:
                cfg[name][hp_name] = hps[hp_name]

    return cfg_dict
