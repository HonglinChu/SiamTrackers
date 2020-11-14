# -*- coding: utf-8 -*
import gc
from typing import Dict, List

from loguru import logger
from yacs.config import CfgNode

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler

from siamfcpp.utils import Timer, dist_utils, ensure_dir

from .adaptor_dataset import AdaptorDataset
from .datapipeline import builder as datapipeline_builder
from .sampler import builder as sampler_builder
from .target import builder as target_builder
from .transformer import builder as transformer_builder


def build(task: str, cfg: CfgNode, seed: int = 0) -> DataLoader:
    r"""
    Arguments
    ---------
    task: str
        task name (track|vos)
    cfg: CfgNode
        node name: data
    seed: int
        seed for random
    """

    if task in ["track", "vos"]:
        # build dummy dataset for purpose of dataset setup (e.g. caching path list)
        logger.info("Build dummy AdaptorDataset")
        dummy_py_dataset = AdaptorDataset(
            task,
            cfg,
            num_epochs=cfg.num_epochs,
            nr_image_per_epoch=cfg.nr_image_per_epoch,
            seed=seed,
        )
        logger.info("Read dummy training sample")
        dummy_sample = dummy_py_dataset[0]   # read dummy sample
        del dummy_py_dataset, dummy_sample
        gc.collect(generation=2)
        logger.info("Dummy AdaptorDataset destroyed.")
        # get world size in case of DDP
        world_size = dist_utils.get_world_size()
        # build real dataset
        logger.info("Build real AdaptorDataset")
        py_dataset = AdaptorDataset(task,
                                    cfg,
                                    num_epochs=cfg.num_epochs,
                                    nr_image_per_epoch=cfg.nr_image_per_epoch)
        # use DistributedSampler in case of DDP
        if world_size > 1:
            py_sampler = DistributedSampler(py_dataset)
            logger.info("Use dist.DistributedSampler, world_size=%d" %
                        world_size)
        else:
            py_sampler = None
        # build real dataloader
        dataloader = DataLoader(
            py_dataset,
            batch_size=cfg.minibatch // world_size,
            shuffle=False,
            pin_memory=cfg.pin_memory,
            num_workers=cfg.num_workers // world_size,
            drop_last=True,
            sampler=py_sampler,
        )
    return dataloader


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
        cfg["exp_name"] = ""
        cfg["exp_save"] = "snapshots"
        cfg["num_epochs"] = 1
        cfg["minibatch"] = 32
        cfg["num_workers"] = 4
        cfg["nr_image_per_epoch"] = 150000
        cfg["pin_memory"] = True
        cfg["datapipeline"] = datapipeline_builder.get_config(task_list)[task]
        cfg["sampler"] = sampler_builder.get_config(task_list)[task]
        cfg["transformer"] = transformer_builder.get_config(task_list)[task]
        cfg["target"] = target_builder.get_config(task_list)[task]

    return cfg_dict
