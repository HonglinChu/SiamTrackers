# -*- coding: utf-8 -*
import os
import os.path as osp
from copy import deepcopy
from typing import Dict, List, Tuple

from loguru import logger

import torch
from torch import nn
from torch.utils.data import DataLoader

from siamfcpp.model.module_base import ModuleBase
from siamfcpp.optim.optimizer.optimizer_base import OptimizerBase
from siamfcpp.utils import Registry, ensure_dir, unwrap_model

TRACK_TRAINERS = Registry('TRACK_TRAINERS')
VOS_TRAINERS = Registry('VOS_TRAINERS')

TASK_TRAINERS = dict(
    track=TRACK_TRAINERS,
    vos=VOS_TRAINERS,
)


class TrainerBase:
    r"""
    Trainer base class (e.g. procedure defined for tracker / segmenter / etc.)
    Interface descriptions:
    """
    # Define your default hyper-parameters here in your sub-class.
    default_hyper_params = dict(
        exp_name="default_training",
        exp_save="snapshots",
        max_epoch=20,
    )

    def __init__(self, optimizer, dataloader, monitors=[]):
        self._hyper_params = deepcopy(
            self.default_hyper_params)  # mapping-like object
        self._state = dict()  # pipeline state
        self._model = optimizer._model
        self._losses = optimizer._model.loss
        self._optimizer = optimizer
        self._monitors = monitors
        self._dataloader = iter(dataloader)  # get the iterabel dataloader

    def get_hps(self) -> Dict:
        r"""
        Getter function for hyper-parameters

        Returns
        -------
        dict
            hyper-parameters
        """
        return self._hyper_params

    def set_hps(self, hps: Dict) -> None:
        r"""
        Set hyper-parameters

        Arguments
        ---------
        hps: dict
            dict of hyper-parameters, the keys must in self.__hyper_params__
        """
        for key in hps:
            if key not in self._hyper_params:
                raise KeyError
            self._hyper_params[key] = hps[key]

    def update_params(self, ):
        self._hyper_params["num_iterations"] = self._hyper_params[
            "nr_image_per_epoch"] // self._hyper_params["minibatch"]
        self._state["snapshot_dir"] = osp.join(self._hyper_params["exp_save"],
                                               self._hyper_params["exp_name"])

        self._state["snapshot_file"] = self._hyper_params["snapshot"]

    def init_train(self):
        r"""
        an interface to process pre-train overhead before training
        trainer's implementation is responsible for checking and 
            calling it automatically before training starts.
        """
        for monitor in self._monitors:
            monitor.init(self._state)

    def train(self):
        r"""
        an interface to train for one epoch
        """
    def set_dataloader(self, dataloader: DataLoader):
        r""""""
        self._dataloader = dataloader

    def set_optimizer(self, optimizer: OptimizerBase):
        r""""""
        self._optimizer = optimizer

    def is_completed(self):
        r"""Return completion status"""
        is_completed = (self._state["epoch"] + 1 >=
                        self._hyper_params["max_epoch"])
        return is_completed

    def load_snapshot(self):
        r""" 
        load snapshot based on self._hyper_params["snapshot"] or self._state["epoch"]
        """
        snapshot_file = self._state["snapshot_file"]
        if osp.exists(snapshot_file):
            dev = self._state["devices"][0]
            snapshot = torch.load(snapshot_file, map_location=dev)
            self._model.load_state_dict(snapshot["model_state_dict"])
            self._optimizer.load_state_dict(snapshot["optimizer_state_dict"])
            self._state["epoch"] = snapshot["epoch"]
            logger.info("Load snapshot from: %s" % osp.realpath(snapshot_file))
        else:
            logger.info("%s does not exist, no snapshot loaded." %
                        snapshot_file)

        logger.info("Train from epoch %d" % (self._state["epoch"] + 1))

    def save_snapshot(self, ):
        r""" 
        save snapshot for current epoch
        """
        epoch = self._state["epoch"]
        snapshot_dir, snapshot_file = self._infer_snapshot_dir_file_from_epoch(
            epoch)
        snapshot_dict = {
            'epoch': epoch,
            'model_state_dict': unwrap_model(self._model).state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict()
        }
        ensure_dir(snapshot_dir)
        torch.save(snapshot_dict, snapshot_file)
        while not osp.exists(snapshot_file):
            logger.info("retrying")
            torch.save(snapshot_dict, snapshot_file)
        logger.info("Snapshot saved at: %s" % snapshot_file)

    def _infer_snapshot_dir_file_from_epoch(self,
                                            epoch: int) -> Tuple[str, str]:
        r"""Infer snapshot's directory & file path based on self._state & epoch number pased in

        Parameters
        ----------
        epoch : int
            epoch number
        
        Returns
        -------
        Tuple[str, str]
            directory and snapshot file
            dir, path
        """
        snapshot_dir = self._state["snapshot_dir"]
        snapshot_file = osp.join(snapshot_dir, "epoch-{}.pkl".format(epoch))
        return snapshot_dir, snapshot_file

    def _get_latest_model_path(self):
        file_dir = self._state["snapshot_dir"]
        file_list = os.listdir(file_dir)
        file_list = [
            file_name for file_name in file_list if file_name.endswith("pkl")
        ]
        if not file_list:
            return "none"
        file_list.sort(key=lambda fn: os.path.getmtime(osp.join(file_dir, fn))
                       if not os.path.isdir(osp.join(file_dir, fn)) else 0)
        return osp.join(file_dir, file_list[-1])

    def resume(self, resume):
        r"""Apply resuming by setting self._state["snapshot_file"]
        Priviledge snapshot_file to epoch number

        Parameters
        ----------
        resume :str
            latest epoch number, by default -1, "latest" or model path
        """
        if resume.isdigit():
            _, snapshot_file = self._infer_snapshot_dir_file_from_epoch(resume)
            self._state["snapshot_file"] = snapshot_file
        elif resume == "latest":
            self._state["snapshot_file"] = self._get_latest_model_path()
        else:
            self._state["snapshot_file"] = resume

    def set_device(self, devs: List[str]):
        self._state["devices"] = [torch.device(dev) for dev in devs]
