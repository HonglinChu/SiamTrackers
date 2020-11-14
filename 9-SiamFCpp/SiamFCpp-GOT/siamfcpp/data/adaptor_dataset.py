# -*- coding: utf-8 -*
from itertools import chain

from loguru import logger

import torch
import torch.multiprocessing
from torch.utils.data import DataLoader, Dataset

from siamfcpp.utils.misc import Timer

from .datapipeline import builder as datapipeline_builder

# pytorch wrapper for multiprocessing
# https://pytorch.org/docs/stable/multiprocessing.html#strategy-management
_SHARING_STRATETY = "file_system"
if _SHARING_STRATETY in torch.multiprocessing.get_all_sharing_strategies():
    torch.multiprocessing.set_sharing_strategy(_SHARING_STRATETY)


class AdaptorDataset(Dataset):
    _EXT_SEED_STEP = 30011  # better to be a prime number
    _SEED_STEP = 10007  # better to be a prime number
    _SEED_DIVIDER = 1000003  # better to be a prime number

    def __init__(
            self,
            task,
            cfg,
            num_epochs=1,
            nr_image_per_epoch=1,
            seed: int = 0,
    ):
        self.datapipeline = None
        self.task = task
        self.cfg = cfg
        self.num_epochs = num_epochs
        self.nr_image_per_epoch = nr_image_per_epoch
        self.ext_seed = seed

    def __getitem__(self, item):
        if self.datapipeline is None:
            # build datapipeline with random seed the first time when __getitem__ is called
            # usually, dataset is already spawned (into subprocess) at this point.
            seed = (torch.initial_seed() + item * self._SEED_STEP +
                    self.ext_seed * self._EXT_SEED_STEP) % self._SEED_DIVIDER
            self.datapipeline = datapipeline_builder.build(self.task,
                                                           self.cfg,
                                                           seed=seed)
            logger.info("AdaptorDataset #%d built datapipeline with seed=%d" %
                        (item, seed))

        training_data = self.datapipeline[item]

        return training_data

    def __len__(self):
        return self.nr_image_per_epoch * self.num_epochs
