# -*- coding: utf-8 -*
import copy
import itertools
from collections import OrderedDict

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

from siamfcpp.model.module_base import ModuleBase
from siamfcpp.optim.optimizer.optimizer_base import OptimizerBase
from siamfcpp.utils import (Timer, ensure_dir, move_data_to_device,
                                unwrap_model)

from ..trainer_base import TRACK_TRAINERS, TrainerBase


@TRACK_TRAINERS.register
class RegularTrainer(TrainerBase):
    r"""
    Trainer to test the vot dataset, the result is saved as follows
    exp_dir/logs/$dataset_name$/$tracker_name$/baseline
                                    |-$video_name$/ floder of result files
                                    |-eval_result.csv evaluation result file

    Hyper-parameters
    ------------------
    devices: List[str]
        list of string
    """
    extra_hyper_params = dict(
        minibatch=1,
        nr_image_per_epoch=1,
        max_epoch=1,
        snapshot="",
    )

    def __init__(self, optimizer, dataloader, monitors=[]):
        r"""
        Crete tester with config and pipeline

        Arguments
        ---------
        optimizer: ModuleBase
            including optimizer, model and loss
        dataloder: DataLoader
            PyTorch dataloader object. 
            Usage: batch_data = next(dataloader)
        """
        super(RegularTrainer, self).__init__(optimizer, dataloader, monitors)
        # update state
        self._state["epoch"] = -1  # uninitialized
        self._state["initialized"] = False
        self._state["devices"] = torch.device("cuda:0")

    def init_train(self, ):
        torch.cuda.empty_cache()
        # move model & loss to target devices
        devs = self._state["devices"]
        self._model.train()
        # load from self._state["snapshot_file"]
        self.load_snapshot()
        # parallelism with Data Parallel (DP)
        if len(self._state["devices"]) > 1:
            self._model = nn.DataParallel(self._model, device_ids=devs)
            logger.info("Use nn.DataParallel for data parallelism")
        super(RegularTrainer, self).init_train()
        logger.info("{} initialized".format(type(self).__name__))

    def train(self):
        if not self._state["initialized"]:
            self.init_train()
        self._state["initialized"] = True

        self._state["epoch"] += 1
        epoch = self._state["epoch"]
        num_iterations = self._hyper_params["num_iterations"]

        # udpate engine_state
        self._state["max_epoch"] = self._hyper_params["max_epoch"]
        self._state["max_iteration"] = num_iterations # 12500

        self._optimizer.modify_grad(epoch)
        pbar = tqdm(range(num_iterations))
        self._state["pbar"] = pbar
        self._state["print_str"] = ""

        time_dict = OrderedDict()
        for iteration, _ in enumerate(pbar):
            self._state["iteration"] = iteration
            with Timer(name="data", output_dict=time_dict):
                training_data = next(self._dataloader)
            training_data = move_data_to_device(training_data,
                                                self._state["devices"][0])

            schedule_info = self._optimizer.schedule(epoch, iteration)
            self._optimizer.zero_grad()

            # forward propagation
            with Timer(name="fwd", output_dict=time_dict):
                predict_data = self._model(training_data)
                training_losses, extras = OrderedDict(), OrderedDict()
                for loss_name, loss in self._losses.items():
                    training_losses[loss_name], extras[loss_name] = loss(
                        predict_data, training_data)
                total_loss = sum(training_losses.values())

            # backward propagation
            with Timer(name="bwd", output_dict=time_dict):
                total_loss.backward()
            self._optimizer.modify_grad(epoch, iteration)
            with Timer(name="optim", output_dict=time_dict):
                self._optimizer.step()

            trainer_data = dict(
                schedule_info=schedule_info,
                training_losses=training_losses,
                extras=extras,
                time_dict=time_dict,
            )

            for monitor in self._monitors:
                monitor.update(trainer_data)
            del training_data
            print_str = self._state["print_str"]
            pbar.set_description(print_str)


RegularTrainer.default_hyper_params = copy.deepcopy(
    RegularTrainer.default_hyper_params)
RegularTrainer.default_hyper_params.update(RegularTrainer.extra_hyper_params)
