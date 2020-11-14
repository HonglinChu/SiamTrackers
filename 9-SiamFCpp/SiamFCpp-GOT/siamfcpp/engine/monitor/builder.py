# -*- coding: utf-8 -*
from typing import Dict, List

from yacs.config import CfgNode

from siamfcpp.utils.misc import merge_cfg_into_hps

from .monitor_base import TASK_MONITORS, MonitorBase


def build(task: str, cfg: CfgNode) -> List[MonitorBase]:
    r"""
    Builder function.

    Arguments
    ---------
    task: str
        builder task name (track|vos)
    cfg: CfgNode
        node name: monitors
    
    Returns
    -------
    List[MonitorBase]
        list of monitors
    """
    assert task in TASK_MONITORS, "no tester for task {}".format(task)
    modules = TASK_MONITORS[task]

    names = cfg.names
    monitors = []
    for name in names:
        monitor = modules[name]()
        hps = monitor.get_hps()
        hps = merge_cfg_into_hps(cfg[name], hps)
        monitor.set_hps(hps)
        monitor.update_params()
        monitors.append(monitor)

    return monitors


def get_config(task_list) -> Dict[str, CfgNode]:
    cfg_dict = {name: CfgNode() for name in task_list}

    for cfg_name, modules in TASK_MONITORS.items():
        cfg = cfg_dict[cfg_name]
        cfg["names"] = [""]

        for name in modules:
            cfg[name] = CfgNode()
            module = modules[name]
            hps = module.default_hyper_params
            for hp_name in hps:
                cfg[name][hp_name] = hps[hp_name]

    return cfg_dict
