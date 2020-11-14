# -*- coding: utf-8 -*-

import argparse
import os.path as osp
import os
import sys
sys.path.append(os.getcwd())

import cv2
from loguru import logger

import torch
import numpy as np

from siamfcpp.config.config import cfg as root_cfg
from siamfcpp.config.config import specify_task
from siamfcpp.data import builder as dataloader_builder
from siamfcpp.data.datapipeline import builder as datapipeline_builder
from siamfcpp.data.dataset import builder as dataset_buidler
from siamfcpp.data.utils.visualization import show_img_FCOS
from siamfcpp.engine import builder as engine_builder
from siamfcpp.model import builder as model_builder
from siamfcpp.model.loss import builder as losses_builder
from siamfcpp.optim import builder as optim_builder
from siamfcpp.pipeline import builder as pipeline_builder
from siamfcpp.utils import Timer, complete_path_wt_root_in_cfg, ensure_dir,load_image

cv2.setNumThreads(1)

# torch.backends.cudnn.enabled = False

# pytorch reproducibility
# https://pytorch.org/docs/stable/notes/randomness.html#cudnn
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def make_parser():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument(
        '--config',
        default='models/siamfcpp/data/siamfcpp_data-trn.yaml',
        type=str,
        help='path to experiment configuration')

    parser.add_argument(
        '--target',
        default='',
        type=str,
        help='targeted debugging module (dataloder|datasampler|dataset))')

    return parser


def scan_key():
    logger.info("Key usage prompt: press ESC for exit")
    key = cv2.waitKey(0) & 0xFF
    if key == 27:
        logger.info("ESC pressed, debugging terminated.")
        exit(0)


if __name__ == '__main__':
    # parsing
    parser = make_parser()
    parsed_args = parser.parse_args()
    # experiment config
    exp_cfg_path = osp.realpath(parsed_args.config)
    root_cfg.merge_from_file(exp_cfg_path)
    logger.info("Load experiment configuration at: %s" % exp_cfg_path)
    logger.info(
        "Merged with root_cfg imported from siamfcpp.config.config.cfg")
    # resolve config
    root_cfg = complete_path_wt_root_in_cfg(root_cfg, ROOT_PATH)
    root_cfg = root_cfg.train
    task, task_cfg = specify_task(root_cfg)
    task_cfg.data.num_workers = 2
    task_cfg.data.sampler.submodules.dataset.GOT10kDataset.check_integrity = False
    task_cfg.freeze()

    if parsed_args.target == "dataloader":
        logger.info("visualize for dataloader")
        with Timer(name="Dataloader building", verbose=True):
            dataloader = dataloader_builder.build(task, task_cfg.data)

        for batch_training_data in dataloader:
            keys = list(batch_training_data.keys())
            batch_size = len(batch_training_data[keys[0]])
            training_samples = [{
                k: v[[idx]]
                for k, v in batch_training_data.items()
            } for idx in range(batch_size)]
            for training_sample in training_samples:
                target_cfg = task_cfg.data.target
                show_img_FCOS(target_cfg[target_cfg.name], training_sample)
                scan_key()
    elif parsed_args.target == "dataset":
        logger.info("visualize for dataset")
       
        datasets = dataset_buidler.build(
            task, task_cfg.data.sampler.submodules.dataset)
        dataset = datasets[0]
        while True:
            # pick a frame randomly
            seq_idx = np.random.choice(range(len(dataset)))
            seq_idx = int(seq_idx)
            seq = dataset[seq_idx]
            # video dataset
            if len(seq['image']) > 1:
                frame_idx = np.random.choice(range(len(seq['image'])))
                frame = {k: seq[k][frame_idx] for k in seq}
                # fetch & visualize data
                im = load_image(frame['image'])
                anno = frame['anno']
            # static image dataset
            else:
                im = load_image(seq['image'][0])
                num_anno = len(seq['anno'])
                if num_anno <= 0:
                    logger.info("no annotation")
                    continue
                anno_idx = np.random.choice(num_anno)
                anno = seq['anno'][anno_idx]
            cv2.rectangle(im,
                          tuple(map(int, anno[:2])),
                          tuple(map(int, anno[2:])), (0, 255, 0),
                          thickness=3)
            im = cv2.resize(im, (0, 0), fx=0.33, fy=0.33)
            cv2.imshow("im", im)
            scan_key()
    elif parsed_args.target == "datapipeline":
        logger.info("visualize for datapipeline")
        datapipeline = datapipeline_builder.build(task, task_cfg.data, seed=1)
        target_cfg = task_cfg.data.target
        while True:
            sampled_data = datapipeline[0]
            print(sampled_data.keys())
            show_img_FCOS(
                target_cfg[target_cfg.name],
                sampled_data,
            )
            scan_key()
    else:
        logger.info("--target {} has not been implemented. ".format(
            parsed_args.target))
