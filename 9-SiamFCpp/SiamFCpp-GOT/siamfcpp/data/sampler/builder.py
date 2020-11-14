# -*- coding: utf-8 -*-

from typing import Dict, List

from yacs.config import CfgNode

from siamfcpp.utils import merge_cfg_into_hps

from ..dataset import builder as dataset_builder
from ..filter import builder as filter_builder
from .sampler_base import TASK_SAMPLERS, DatasetBase


def build(task: str, cfg: CfgNode, seed: int = 0) -> DatasetBase:
    r"""
    Arguments
    ---------
    task: str
        task name (track|vos)
    cfg: CfgNode
        node name: sampler
    seed: int
        seed for rng initialization
    """
    assert task in TASK_SAMPLERS, "invalid task name"
    MODULES = TASK_SAMPLERS[task]

    submodules_cfg = cfg.submodules

    dataset_cfg = submodules_cfg.dataset
    datasets = dataset_builder.build(task, dataset_cfg)

    if submodules_cfg.filter.name != "":
        filter_cfg = submodules_cfg.filter
        data_filter = filter_builder.build(task, filter_cfg)
    else:
        data_filter = None

    name = cfg.name
    module = MODULES[name](datasets, seed=seed, data_filter=data_filter)

    hps = module.get_hps()
    hps = merge_cfg_into_hps(cfg[name], hps)
    module.set_hps(hps)
    module.update_params()

    return module


def get_config(task_list: List) -> Dict[str, CfgNode]:
    cfg_dict = {name: CfgNode() for name in task_list}

    for cfg_name, modules in TASK_SAMPLERS.items():
        cfg = cfg_dict[cfg_name]
        cfg["name"] = ""

        for name in modules:
            cfg[name] = CfgNode()
            module = modules[name]
            hps = module.default_hyper_params
            for hp_name in hps:
                cfg[name][hp_name] = hps[hp_name]

        cfg["submodules"] = CfgNode()
        cfg["submodules"]["dataset"] = dataset_builder.get_config(
            task_list)[cfg_name]
        cfg["submodules"]["filter"] = filter_builder.get_config(
            task_list)[cfg_name]

    return cfg_dict
