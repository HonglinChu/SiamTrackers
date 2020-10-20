# -*- coding: utf-8 -*
from collections import OrderedDict
from typing import Dict, List

from loguru import logger
from yacs.config import CfgNode

from siamfcpp.model.loss.loss_base import TASK_LOSSES
from siamfcpp.utils.misc import merge_cfg_into_hps


def build(task: str, cfg: CfgNode):
    if task in TASK_LOSSES:
        MODULES = TASK_LOSSES[task]
    else:
        logger.error("no loss for task {}".format(task))
        exit(-1)

    names = cfg.names
    loss_dict = OrderedDict()
    for name in names:
        assert name in MODULES, "loss {} not registered for {}!".format(
            name, task)
        module = MODULES[name]()
        hps = module.get_hps()
        hps = merge_cfg_into_hps(cfg[name], hps)
        module.set_hps(hps)
        module.update_params()
        loss_dict[cfg[name].name] = module

    return loss_dict


def get_config(task_list: List) -> Dict[str, CfgNode]:
    cfg_dict = {name: CfgNode() for name in task_list}

    for cfg_name, modules in TASK_LOSSES.items():
        cfg = cfg_dict[cfg_name]
        cfg["names"] = list()
        for name in modules:
            cfg[name] = CfgNode()
            backbone = modules[name]
            hps = backbone.default_hyper_params
            for hp_name in hps:
                cfg[name][hp_name] = hps[hp_name]
    return cfg_dict
