# -*- coding: utf-8 -*-
from paths import ROOT_PATH  # isort:skip
# ROOT_PATH=/home/ubuntu/pytorch/pytorch-tracking/SiamFC++
import os
import argparse
from loguru import logger
import torch

from siamfcpp.config.config import cfg as root_cfg
from siamfcpp.config.config import specify_task         # sot or vos
from siamfcpp.engine.builder import build as tester_builder
from siamfcpp.model import builder as model_builder
from siamfcpp.pipeline import builder as pipeline_builder
from siamfcpp.utils import complete_path_wt_root_in_cfg

# def make_parser():
#     parser = argparse.ArgumentParser(description='Test')
#     parser.add_argument('-cfg',
#                         '--config',
#                         default='',
#                         type=str,
#                         help='experiment configuration')
#     return parser

# linlin  2020-05-27
# def build_sat_tester(task_cfg):
#     # build model
#     tracker_model = model_builder.build("track", task_cfg.tracker_model)
#     tracker = pipeline_builder.build("track",
#                                      task_cfg.tracker_pipeline,
#                                      model=tracker_model)
#     segmenter = model_builder.build('vos', task_cfg.segmenter)
#     # build pipeline
#     pipeline = pipeline_builder.build('vos',
#                                       task_cfg.pipeline,
#                                       segmenter=segmenter,
#                                       tracker=tracker)
#     # build tester
#     testers = tester_builder('vos', task_cfg.tester, "tester", pipeline)
#     return testers

model=2   # 
if model==1:
    config='experiments/siamfcpp/test/vot/siamfcpp_alexnet.yaml'    #VOT
elif model==2:
    config='experiments/siamfcpp/test/otb/siamfcpp_alexnet-otb.yaml'#OTB


#siamfc++的测试代码
def build_siamfcpp_tester(task_cfg):
    # build model
    model = model_builder.build("track", task_cfg.model)
    # build pipeline
    pipeline = pipeline_builder.build("track", task_cfg.pipeline, model)
    # build tester
    testers = tester_builder("track", task_cfg.tester, "tester", pipeline)
    return testers
    
if __name__ == '__main__':
    # parsing
    #parser = make_parser()
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('-cfg', '--config', default=config,type=str,help='experiment configuration')
    parsed_args = parser.parse_args()

    # experiment config
    exp_cfg_path = os.path.realpath(parsed_args.config)
    root_cfg.merge_from_file(exp_cfg_path)
    logger.info("Load experiment configuration at: %s" % exp_cfg_path)

    # resolve config
    root_cfg = complete_path_wt_root_in_cfg(root_cfg, ROOT_PATH)
    root_cfg = root_cfg.test
    task, task_cfg = specify_task(root_cfg)
    task_cfg.freeze()

    torch.multiprocessing.set_start_method('spawn', force=True)

    if task == 'track':
        testers = build_siamfcpp_tester(task_cfg)

    #linlin 2020-05-27
    # elif task == 'vos':
    #     testers = build_sat_tester(task_cfg)

    for tester in testers:
        tester.test()
