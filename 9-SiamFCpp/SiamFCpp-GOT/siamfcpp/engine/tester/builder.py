# -*- coding: utf-8 -*
from typing import Dict, List

from loguru import logger
from yacs.config import CfgNode

from siamfcpp.pipeline.pipeline_base import PipelineBase
from siamfcpp.utils import merge_cfg_into_hps

from .tester_base import TASK_TESTERS


def build(task: str, cfg: CfgNode, pipeline: PipelineBase):
    r"""
    Builder function.

    Arguments
    ---------
    task: str
        builder task name (track|vos)
    cfg: CfgNode
        buidler configuration, 
        node nams: tester

    Returns
    -------
    TesterBase
        tester built by builder
    """
    assert task in TASK_TESTERS, "no tester for task {}".format(task)
    MODULES = TASK_TESTERS[task]

    names = cfg.names
    testers = []
    # tester for multiple experiments
    for name in names:
        tester = MODULES[name](pipeline)
        hps = tester.get_hps()
        hps = merge_cfg_into_hps(cfg[name], hps)
        tester.set_hps(hps)
        tester.update_params()
        testers.append(tester)
    return testers


def get_config(task_list: List) -> Dict[str, CfgNode]:
    r"""
    Get available component list config

    Returns
    -------
    Dict[str, CfgNode]
        config with list of available components
    """
    cfg_dict = {name: CfgNode() for name in task_list}

    for cfg_name, MODULES in TASK_TESTERS.items():
        cfg = cfg_dict[cfg_name]
        cfg["names"] = []
        for name in MODULES:
            cfg["names"].append(name)
            cfg[name] = CfgNode()
            tester = MODULES[name]
            hps = tester.default_hyper_params
            for hp_name in hps:
                cfg[name][hp_name] = hps[hp_name]
    return cfg_dict
