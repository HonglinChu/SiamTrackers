# -*- coding: utf-8 -*
from typing import Dict, List

from loguru import logger
from yacs.config import CfgNode

from siamfcpp.data import builder as dataloder_builder
from siamfcpp.model import builder as model_builder
from siamfcpp.model.loss import builder as loss_builder
from siamfcpp.optim.optimizer import builder as optimizer_builder
from siamfcpp.utils.misc import merge_cfg_into_hps

from ..monitor import builder as monitor_builder
from ..monitor.monitor_base import TASK_MONITORS
from .trainer_base import TASK_TRAINERS, TrainerBase


def build(task: str, cfg: CfgNode, optimizer, dataloader,
          tracker=None) -> TrainerBase:
    r"""
    Builder function.

    Arguments
    ---------
    task: str
        builder task name (track|vos)
    cfg: CfgNode
        node name: trainer

    Returns
    -------
    TrainerBase
        tester built by builder
    """
    assert task in TASK_TRAINERS, "no trainer for task {}".format(task)
    MODULES = TASK_TRAINERS[task]

    # build monitors
    if "monitors" in cfg:
        monitor_cfg = cfg.monitors
        monitors = monitor_builder.build(task, monitor_cfg)
    else:
        monitors = []

    name = cfg.name
    if task == "vos":
        trainer = MODULES[name](optimizer, dataloader, monitors, tracker)
    else:
        trainer = MODULES[name](optimizer, dataloader, monitors)
    hps = trainer.get_hps()
    hps = merge_cfg_into_hps(cfg[name], hps)
    trainer.set_hps(hps)
    trainer.update_params()

    return trainer


def get_config(task_list: List) -> Dict[str, CfgNode]:
    r"""
    Get available component list config

    Returns
    -------
    Dict[str, CfgNode]
        config with list of available components
    """
    cfg_dict = {name: CfgNode() for name in task_list}

    for cfg_name, MODULES in TASK_TRAINERS.items():
        cfg = cfg_dict[cfg_name]
        cfg["name"] = ""

        for name in MODULES:
            cfg[name] = CfgNode()
            module = MODULES[name]
            hps = module.default_hyper_params
            for hp_name in hps:
                cfg[name][hp_name] = hps[hp_name]

        cfg["monitors"] = monitor_builder.get_config(task_list)[cfg_name]

    return cfg_dict
