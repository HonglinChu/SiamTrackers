# -*- coding: utf-8 -*
from typing import Dict, List

from loguru import logger
from yacs.config import CfgNode

from torch import nn

from .backbone import builder as backbone_builder
from .loss import builder as loss_builder
from .task_head import builder as head_builder
from .task_model import builder as task_builder


def build(
        task: str,
        cfg: CfgNode,
):
    r"""
    Builder function.

    Arguments
    ---------
    task: str
        builder task name (track|vos)
    cfg: CfgNode
        node name: model

    Returns
    -------
    torch.nn.Module
        module built by builder
    """
    if task == "track":
        backbone = backbone_builder.build(task, cfg.backbone)#AlexNet
        head = head_builder.build(task, cfg.task_head)# 
        losses = loss_builder.build(task, cfg.losses)
        task_model = task_builder.build(task, cfg.task_model, backbone, head,
                                        losses)

    elif task == "vos":
        gml_extractor = backbone_builder.build(task, cfg.gml_extractor)
        encoder_basemodel = backbone_builder.build(task, cfg.encoder_basemodel)
        joint_encoder = backbone_builder.build(task, cfg.encoder,
                                               encoder_basemodel)
        decoder = head_builder.build(task, cfg.task_head)
        losses = loss_builder.build(task, cfg.losses)
        task_model = task_builder.build_sat_model(task,
                                                  cfg.task_model,
                                                  gml_extractor=gml_extractor,
                                                  joint_encoder=joint_encoder,
                                                  decoder=decoder,
                                                  loss=losses)

    else:
        logger.error("model for task {} has not been implemented".format(task))
        exit(-1)
    if cfg.use_sync_bn:
        task_model = nn.SyncBatchNorm.convert_sync_batchnorm(task_model)
    return task_model


def get_config(task_list: List) -> Dict[str, CfgNode]:
    r"""
    Get available component list config

    Returns
    -------
    Dict[str, CfgNode]
        config with list of available components
    """
    cfg_dict = {task: CfgNode() for task in task_list}

    for task in cfg_dict:
        cfg = cfg_dict[task]

        if task == "vos":
            cfg["basemodel_target"] = backbone_builder.get_config(
                task_list)[task]
            cfg["basemodel_search"] = backbone_builder.get_config(
                task_list)[task]
            cfg["encoder_basemodel"] = backbone_builder.get_config(
                task_list)[task]
            cfg["encoder"] = backbone_builder.get_config(task_list)[task]
            cfg["gml_extractor"] = backbone_builder.get_config(task_list)[task]
        cfg["backbone"] = backbone_builder.get_config(task_list)[task]
        cfg["task_head"] = head_builder.get_config(task_list)[task]
        cfg["losses"] = loss_builder.get_config(task_list)[task]
        cfg["task_model"] = task_builder.get_config(task_list)[task]
        cfg["use_sync_bn"] = False

    return cfg_dict
