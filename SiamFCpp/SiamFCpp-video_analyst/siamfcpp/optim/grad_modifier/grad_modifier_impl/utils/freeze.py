# -*- coding: utf-8 -*

import re
from collections import OrderedDict
from typing import Dict, List

from loguru import logger
from yacs.config import CfgNode

import torch
from torch import nn


class FreezeStateMonitor:
    """ Monitor the freezing state continuously and print """
    def __init__(self, module: nn.Module, verbose=True):
        """
        :param module: module to be monitored
        :param verbose:
        """
        self.module = module
        self.verbose = verbose

    def __enter__(self, ):
        self.old_freeze_state = OrderedDict([
            (k, v.requires_grad) for k, v in self.module.named_parameters()
        ])

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.new_freeze_state = OrderedDict([
            (k, v.requires_grad) for k, v in self.module.named_parameters()
        ])
        if self.verbose:
            assert set(list(self.new_freeze_state.keys())) == set(
                list(self.old_freeze_state.keys()))
            any_change = False
            for k in self.new_freeze_state.keys():
                change = (self.old_freeze_state[k] != self.new_freeze_state[k])
                if change:
                    logger.info(k, "changed:", self.old_freeze_state[k], "->",
                                self.new_freeze_state[k])
                any_change = any_change or change


def dynamic_freeze(module: nn.Module,
                   compiled_regex=re.compile(".*"),
                   requires_grad: bool = False,
                   verbose: bool = False):
    """Perform dynamic freezing
    
    Parameters
    ----------
    module : [type]
        [description]
    compiled_regex : [type], optional
        compiled regular expression, by default re.compile(".*")
    requires_grad : bool, optional
        [description], by default False
    verbose : bool, optional
        [description], by default False
    """
    with FreezeStateMonitor(module, verbose=verbose):
        for k, v in module.named_parameters():
            if (compiled_regex.search(k) is not None):
                v.requires_grad = requires_grad


# def apply_freeze_schedule(module, epoch, schedule_list, verbose=True):
#     with FreezeStateMonitor(module, verbose=verbose):
#         for param_filter, requires_grad_cond in schedule_list:
#             dynamic_freeze(module,
#                            param_filter=param_filter,
#                            requires_grad=requires_grad_cond(epoch))


def apply_freeze_schedule(module: nn.Module,
                          epoch: int,
                          schedule: List[Dict],
                          verbose: bool = True):
    r"""
    Apply dynamic freezing schedule with verbose
    
    Arguments:
    module: nn.Module
        model to be scheduled
    epoch: int
        current epoch
    schedules: List[Dict]
        lsit of schedule
        schedule: Dict
            "regex": regex to filter parameters
            "epoch": epoch where the schedule starts from
            "freezed": freeze or not

    """
    with FreezeStateMonitor(module, verbose=verbose):
        for freeze_action in schedule:
            # param_filter, requires_grad_cond
            compiled_regex = freeze_action["compiled_regex"]
            requires_grad = (
                (epoch >= freeze_action["epoch"]) != freeze_action["freezed"]
            )  # XOR
            dynamic_freeze(module,
                           compiled_regex=compiled_regex,
                           requires_grad=requires_grad)
