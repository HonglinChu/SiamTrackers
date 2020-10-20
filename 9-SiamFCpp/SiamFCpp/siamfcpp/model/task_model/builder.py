# -*- coding: utf-8 -*
from typing import Dict, List

from loguru import logger
from yacs.config import CfgNode

from siamfcpp.model.module_base import ModuleBase
from siamfcpp.model.task_model.taskmodel_base import TASK_TASKMODELS
from siamfcpp.utils import merge_cfg_into_hps


def build(task: str,
          cfg: CfgNode,
          backbone: ModuleBase,
          head: ModuleBase,
          loss: ModuleBase = None):
    r"""
    Builder function.

    Arguments
    ---------
    task: str
        builder task name
    cfg: CfgNode
        buidler configuration
    backbone: torch.nn.Module
        backbone used by task module.
    head: torch.nn.Module
        head network used by task module.
    loss: torch.nn.Module
        criterion module used by task module (for training). None in case other than training.

    Returns
    -------
    torch.nn.Module
        task module built by builder
    """
    if task in TASK_TASKMODELS:
        task_modules = TASK_TASKMODELS[task]
    else:
        logger.error("no task model for task {}".format(task))
        exit(-1)

    if task == "track":
        name = cfg.name
        task_module = task_modules[name](backbone, head, loss)
        hps = task_module.get_hps()
        hps = merge_cfg_into_hps(cfg[name], hps)
        task_module.set_hps(hps)
        task_module.update_params()
        return task_module
    else:
        logger.error("task model {} is not completed".format(task))
        exit(-1)


def build_sat_model(task: str,
                    cfg: CfgNode,
                    gml_extractor=None,
                    joint_encoder=None,
                    decoder=None,
                    loss: ModuleBase = None):
    r"""
    Builder function for SAT.

    Arguments
    ---------
    task: str
        builder task name
    cfg: CfgNode
        buidler configuration
    gml_extractor: torch.nn.Module
        feature extractor for global modeling loop
    joint_encoder: torch.nn.Module
        joint encoder
    decoder: torch.nn.Module
        decoder for SAT
    loss: torch.nn.Module
        criterion module used by task module (for training). None in case other than training.

    Returns
    -------
    torch.nn.Module
        task module built by builder
    """

    if task == "vos":
        task_modules = TASK_TASKMODELS[task]
    else:
        logger.error("sat model builder could not build task {}".format(task))
        exit(-1)
    name = cfg.name
    #SatVOS
    task_module = task_modules[name](gml_extractor, joint_encoder, decoder,
                                     loss)
    hps = task_module.get_hps()
    hps = merge_cfg_into_hps(cfg[name], hps)
    task_module.set_hps(hps)
    task_module.update_params()
    return task_module


def get_config(task_list: List) -> Dict[str, CfgNode]:
    """
    Get available component list config

    Returns
    -------
    Dict[str, CfgNode]
        config with list of available components
    """
    cfg_dict = {task: CfgNode() for task in task_list}
    for cfg_name, task_module in TASK_TASKMODELS.items():
        cfg = cfg_dict[cfg_name]
        cfg["name"] = "unknown"
        for name in task_module:
            cfg[name] = CfgNode()
            task_model = task_module[name]
            hps = task_model.default_hyper_params
            for hp_name in hps:
                cfg[name][hp_name] = hps[hp_name]
    return cfg_dict
