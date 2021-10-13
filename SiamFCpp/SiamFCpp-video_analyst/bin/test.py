# -*- coding: utf-8 -*-
import os
import argparse
from loguru import logger
import torch

import sys
sys.path.append(os.getcwd())

from siamfcpp.config.config import cfg as root_cfg
from siamfcpp.config.config import specify_task         # sot or vos
from siamfcpp.engine.builder import build as tester_builder
from siamfcpp.model import builder as model_builder
from siamfcpp.pipeline import builder as pipeline_builder
from siamfcpp.utils import complete_path_wt_root_in_cfg

model=2   # 
if model==1:
    config='models/siamfcpp/test/vot/siamfcpp_alexnet.yaml'    #VOT
elif model==2:
    config='models/siamfcpp/test/otb/siamfcpp_alexnet-otb.yaml'#OTB

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
    ROOT_PATH=os.getcwd()
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
