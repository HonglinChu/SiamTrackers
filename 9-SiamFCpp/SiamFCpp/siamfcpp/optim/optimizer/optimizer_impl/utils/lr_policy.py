# -*- coding: utf-8 -*
"""
Support LR(Learning Rate) scheduling for training
Usage:
In train.py:
    lr_scheduler = BaseLRObject(*args, **kwargs)
    ...
    for epoch in ...:
        for iter in ...:
            lr = lr_scheduler(\epoch, iter)
            ...training process...

Get number of epochs scheduled:
    max_epoch = len(lr_scheduler)

Combination of scheduler: schuler1 for first len(scheduler1) epoch
    # e.g. insert warmingup scheduler before the decaying scheduler
    lr_scheduler_combined = ListLR(scheduler_warmup, scheduler_decay)
or
    lr_scheduler_combined = ListLR(*[scheduler_warmup, scheduler_decay])
or
    listLR1 + listLR2

Visulize scheduled LR
    lr_scheduler = ListLR(LinearLR(start_lr=1e-6, end_lr=1e-1, max_epoch=5, max_iter=5000),
                          ExponentialLR(start_lr=1e-1, end_lr=1e-4, max_epoch=15, max_iter=5000))
    plot_LR(lr_scheduler, 'Exponential decay with warmup')
See the bottom of code for more plot examples, together with some exmples for .yaml configuration files (commented part).
"""
import json
import math
from abc import ABCMeta, abstractmethod
from typing import List

import numpy as np
from yacs.config import CfgNode

from siamfcpp.utils import Registry

__all__ = ["ListLR", "LinearLR", "ExponentialLR", "CosineLR"]

LR_POLICIES = Registry("LR_POLICY")


def build(cfg: List[str], **kwargs):
    r"""
    Build lr scheduler with configuration

    Arguments
    ---------
    cfg: List[str]
        list of JSON string containing lr scheduling
    **kwargs
        extra keyword argument that apply to all schedule

    Returns
    -------
    ListLR

    """
    cfg = [json.loads(c) for c in cfg]

    SingleLRs = []
    for phase_cfg in cfg:
        phase_cfg.update(kwargs)
        policy = LR_POLICIES[phase_cfg["name"]](**phase_cfg)
        SingleLRs.append(policy)

    LR = ListLR(*SingleLRs)

    return LR


class BaseLR():
    __metaclass__ = ABCMeta
    max_iter = 1

    @abstractmethod
    def get_lr(self, epoch=0, iter=0):
        pass

    @property
    @abstractmethod
    def max_epoch(self, epoch, iter=0):
        pass


class ListLR(BaseLR):
    def __init__(self, *args):
        self.LRs = [LR for LR in args]

    def get_lr(self, epoch=0, iter=0):
        for LR in self.LRs:
            if epoch < len(LR):
                break
            else:
                epoch -= len(LR)
        return LR.get_lr(epoch, iter)

    def __add__(self, other):
        if isinstance(other, ListLR):
            self.LRs.extend(other.LRs)
        elif isinstance(other, BaseLR):
            self.LRs.append(ListLR(other))
        else:
            raise TypeError(
                'Argument other must be either ListLR or BaseLR object.')
        return self

    def __len__(self, ):
        return sum([len(LR) for LR in self.LRs])

    @property
    def max_iter(self):
        return max([LR.max_iter for LR in self.LRs])


@LR_POLICIES.register
class MultiStageLR(BaseLR):
    """ Multi-stage learning rate scheduler
    """
    def __init__(self, lr_stages, **kwargs):
        """
        :param lr_stages: list, [(milestone1, lr1), (milestone2, lr2), ...]
        """
        assert type(lr_stages) in [list, tuple] and len(lr_stages[0]) == 2, \
            'lr_stages must be list or tuple, with [iters, lr] format'
        self._lr_stages = lr_stages
        self.max_iter = 1

    def get_lr(self, epoch=0, iter=0):
        for (stage_epoch, lr) in self._lr_stages:
            if 0 <= epoch < stage_epoch:
                return lr
        raise ValueError('Invalid epoch.')

    def __len__(self):
        return self._lr_stages[-1][0]


def _IDENTITY(x):
    return x


class TransitionLR(BaseLR):
    """
    Transition scheduler, to be inheritated for different usage
    Idea: the majority of lr scheduling curve becomes linear function after a inversible mapping

    Formula:
    lr = post_func( 
                                                                               (epoch*max_iter+iter)
    pre_func(start_lr) + (pre_func(end_lr)-pre_func(start_lr)) * trans_func( --------------------------- )
                  )                                                           (max_epoch*max_iter+iter)

    Current descendants: 
    - LinearLR
    - ExponentialLR, 
    - CosineLR

    To create new lr scheduling curve:
        please override pre_func, trans_func, and post_func
    """
    def __init__(self, start_lr=0, end_lr=0, max_epoch=1, max_iter=1, **kwargs):
        self._start_lr = start_lr
        self._end_lr = end_lr
        self._max_epoch = max_epoch
        self._max_iter = max_iter
        self._pre_func = _IDENTITY
        self._trans_func = _IDENTITY
        self._post_func = _IDENTITY

    def get_lr(self, epoch=0, iter=0):
        if not (0 <= epoch < self._max_epoch):
            raise ValueError('Invalid epoch.')
        if not (0 <= iter < self._max_iter):
            raise ValueError('Invalid iter.')
        start_value = self._pre_func(self._start_lr)
        end_value = self._pre_func(self._end_lr)
        trans_ratio = self._trans_func((epoch * self._max_iter + iter) /
                                       (self._max_epoch * self._max_iter))
        value = self._post_func(start_value +
                                (end_value - start_value) * trans_ratio)
        return value

    def __len__(self):
        return self._max_epoch

    @property
    def max_iter(self):
        return self._max_iter


@LR_POLICIES.register
class LinearLR(TransitionLR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pre_func = _IDENTITY
        self._trans_func = _IDENTITY
        self._post_func = _IDENTITY


@LR_POLICIES.register
class ExponentialLR(TransitionLR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pre_func = math.log
        self._trans_func = _IDENTITY
        self._post_func = math.exp


@LR_POLICIES.register
class CosineLR(TransitionLR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pre_func = _IDENTITY
        self._trans_func = self._cosine_curve
        self._post_func = _IDENTITY

    def _cosine_curve(self, x):
        return (1 - math.cos(x * math.pi)) / 2


def plot_LR(LR: BaseLR, title='Untitled'):
    """ plot learning rate scheduling plan of an BaseLR object """
    assert isinstance(LR, BaseLR)
    import itertools
    import matplotlib.pyplot as plt
    max_iter = LR.max_iter
    max_epoch = len(LR)
    epochs = np.arange(0, max_epoch)
    iters = np.arange(0, max_iter, max(max_iter // 10, 1))

    accum_iters = []
    lrs = []
    for epoch, iter in itertools.product(epochs, iters):
        accum_iter = epoch * max_iter + iter
        lr = LR.get_lr(epoch, iter)
        accum_iters.append(accum_iter)
        lrs.append(lr)
    plt.figure()
    plt.plot(accum_iters, lrs)
    plt.xlabel('iterations')
    plt.ylabel('learning rate')
    plt.title('%s learning rate scheduling' % title)
    plt.show()


def schedule_lr(optimizer, lr):
    """ adjust learning rate of a PyTorch optimizer """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


if __name__ == '__main__':

    lr_scheduler = ListLR(
        LinearLR(start_lr=1e-6, end_lr=1e-1, max_epoch=5, max_iter=5000),
        LinearLR(start_lr=1e-1, end_lr=1e-4, max_epoch=15, max_iter=5000))
    plot_LR(lr_scheduler, 'Linear decay with warmup')
    # Example for .yaml configuration file
    # =========
    # lr_policy:
    # - >
    # {
    # "name": "LinearLR",
    # "start_lr": 0.000001,
    # "end_lr": 0.01,
    # "max_epoch": 5
    # }
    # - >
    # {
    # "name": "LinearLR",
    # "start_lr": 0.01,
    # "end_lr": 0.0001,
    # "max_epoch": 15
    # }

    lr_scheduler = ListLR(
        LinearLR(start_lr=1e-6, end_lr=1e-1, max_epoch=5, max_iter=5000),
        ExponentialLR(start_lr=1e-1, end_lr=1e-4, max_epoch=15, max_iter=5000))
    plot_LR(lr_scheduler, 'Exponential decay with warmup')
    # Example for .yaml configuration file
    # =========
    # lr_policy:
    # - >
    # {
    # "name": "LinearLR",
    # "start_lr": 0.000001,
    # "end_lr": 0.1,
    # "max_epoch": 5
    # }
    # - >
    # {
    # "name": "LinearLR",
    # "start_lr": 0.1,
    # "end_lr": 0.0001,
    # "max_epoch": 15
    # }

    lr_scheduler = ListLR(
        LinearLR(start_lr=1e-6, end_lr=8e-2, max_epoch=2, max_iter=1250),
        CosineLR(start_lr=8e-2, end_lr=1e-7, max_epoch=20, max_iter=1250))
    plot_LR(lr_scheduler, 'Cosine annealing with warmup')
    # Example for .yaml configuration file
    # =========
    # lr_policy:
    # - >
    # {
    # "name": "LinearLR",
    # "start_lr": 0.000001,
    # "end_lr": 0.1,
    # "max_epoch": 5
    # }
    # - >
    # {
    # "name": "CosineLR",
    # "start_lr": 0.1,
    # "end_lr": 0.0001,
    # "max_epoch": 15
    # }

    lr_scheduler = ListLR(LinearLR(start_lr=1e-6, end_lr=1e-1, max_epoch=5, max_iter=5000)) + \
                   ListLR(ExponentialLR(start_lr=1e-1, end_lr=1e-4, max_epoch=15, max_iter=5000))
    plot_LR(lr_scheduler, 'Exponential decay with warmup')
    # Example for .yaml configuration file
    # =========
    # lr_policy:
    # - >
    # {
    # "name": "LinearLR",
    # "start_lr": 0.000001,
    # "end_lr": 0.1,
    # "max_epoch": 5
    # }
    # - >
    # {
    # "name": "ExponentialLR",
    # "start_lr": 0.1,
    # "end_lr": 0.0001,
    # "max_epoch": 15
    # }

    lr_scheduler = MultiStageLR([(10, 0.1), (40, 0.01), (70, 0.001),
                                 (120, 0.0001)])
    plot_LR(lr_scheduler, 'Step decay')
    # Example for .yaml configuration file
    # =========
    # lr_policy:
    # ->
    # {
    # "name": "MultiStageLR",
    # "lr_stages": [[10, 0.1], [40, 0.01], [70, 0.001], [120, 0.0001]]
    # }
    # from IPython import embed;embed()
