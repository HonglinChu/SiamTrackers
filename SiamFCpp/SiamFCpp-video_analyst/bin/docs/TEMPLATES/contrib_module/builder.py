# -*- coding: utf-8 -*-
from typing import Dict, List

from yacs.config import CfgNode

from .contrib_module_base import TASK_CONTRIB_MODULES, ContribModuleBase
from videoanalyst.utils import merge_cfg_into_hps


def build(task: str, cfg: CfgNode) -> ContribModule:
    r"""
    Arguments
    ---------
    task: str
        task name (track|vos)
    cfg: CfgNode
        node name: contrib_module
    """
    assert task in TASK_CONTRIB_MODULES, "invalid task name"
    MODULES = TASK_CONTRIB_MODULES[task]

    name = cfg.name
    module = MODULES[name]()

    hps = module.get_hps()
    hps = merge_cfg_into_hps(cfg[name], hps)
    module.set_hps(hps)
    module.update_params()

    return module


def get_config(task_list) -> Dict[str, CfgNode]:
    cfg_dict = {name: CfgNode() for name in task_list}
    for cfg_name, modules in TASK_CONTRIB_MODULES.items():
        cfg = cfg_dict[cfg_name]
        cfg["name"] = ""

        for name in modules:
            cfg[name] = CfgNode()
            module = modules[name]
            hps = module.default_hyper_params
            for hp_name in hps:
                cfg[name][hp_name] = hps[hp_name]

    return cfg_dict
