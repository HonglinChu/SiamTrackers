# -*- coding: utf-8 -*-
import argparse
import os.path as osp
import pickle
import os

import sys
sys.path.append(os.getcwd())

import cv2
from loguru import logger

import torch

from siamfcpp.config.config import cfg as root_cfg
from siamfcpp.config.config import specify_task
from siamfcpp.data import builder as dataloader_builder
from siamfcpp.engine import builder as engine_builder
from siamfcpp.model import builder as model_builder
from siamfcpp.model.loss import builder as losses_builder
from siamfcpp.optim import builder as optim_builder
from siamfcpp.pipeline import builder as pipeline_builder
from siamfcpp.utils import Timer, complete_path_wt_root_in_cfg, ensure_dir

cv2.setNumThreads(1)

# torch.backends.cudnn.enabled = False

# pytorch reproducibility
# https://pytorch.org/docs/stable/notes/randomness.html#cudnn
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# 在工作目录下运行: python3 ./bin/train.py --config 'models/siamfcpp/train/got10k/siamfcpp_alexnet-trn.yaml'

config='models/siamfcpp/train/got10k/siamfcpp_alexnet-trn.yaml'

#config= 'models/siamfcpp/train/got10k/siamfcpp_alexnet-got10k-crop511-trn.yaml'

if __name__ == '__main__':
    # parsing
    #parser = make_parser()
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('-cfg','--config',default=config, type=str,
                        help='path to experiment configuration')
    parser.add_argument( '-r', '--resume', default="",
            help=r"completed epoch's number, latest or one model path")
    parsed_args = parser.parse_args()

    # experiment config
    exp_cfg_path = osp.realpath(parsed_args.config)
    root_cfg.merge_from_file(exp_cfg_path)

    # resolve config\
    ROOT_PATH = os.getcwd()
    root_cfg = complete_path_wt_root_in_cfg(root_cfg, ROOT_PATH)
    root_cfg = root_cfg.train
    task, task_cfg = specify_task(root_cfg)  # task = track
    task_cfg.freeze()

    # log config
    log_dir = osp.join(task_cfg.exp_save, task_cfg.exp_name, "logs")
    ensure_dir(log_dir)
    logger.configure(
        handlers=[
            dict(sink=sys.stderr, level="INFO"),
            dict(sink=osp.join(log_dir, "train_log.txt"),
                 enqueue=True,
                 serialize=True,
                 diagnose=True,
                 backtrace=True,
                 level="INFO")
        ],
        extra={"common_to_all": "default"},
    )

    # backup config
    logger.info("Load experiment configuration at: %s" % exp_cfg_path)

    logger.info("Merged with root_cfg imported from videoanalyst.config.config.cfg")

    cfg_bak_file = osp.join(log_dir, "%s_bak.yaml" % task_cfg.exp_name)

    with open(cfg_bak_file, "w") as f:
        f.write(task_cfg.dump())

    logger.info("Task configuration backed up at %s" % cfg_bak_file)
    
    # device config
    if task_cfg.device == "cuda":
        
        world_size = task_cfg.num_processes

        assert torch.cuda.is_available(), "please check your devices"
        assert torch.cuda.device_count(
        ) >= world_size, "cuda device {} is less than {}".format(
            torch.cuda.device_count(), world_size)

        devs = ["cuda:{}".format(i) for i in range(world_size)]
    else:
        devs = ["cpu"]
    # build model
    model = model_builder.build(task, task_cfg.model)
    model.set_device(devs[0])
    # load data
    with Timer(name="Dataloader building", verbose=True):
        dataloader = dataloader_builder.build(task, task_cfg.data)
    # build optimizer
    optimizer = optim_builder.build(task, task_cfg.optim, model)
    # build trainer
    trainer = engine_builder.build(task, task_cfg.trainer, "trainer", optimizer,
                                   dataloader)
    trainer.set_device(devs)
    trainer.resume(parsed_args.resume)
    # trainer.init_train()
    logger.info("Start training")
    while not trainer.is_completed():
        trainer.train()
        trainer.save_snapshot()
    logger.info("Training completed.")
