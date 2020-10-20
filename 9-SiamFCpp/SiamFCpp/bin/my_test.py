# -*- coding: utf-8 -*-
from paths import ROOT_PATH  # isort:skip

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

model=7 # 
if model==1:
    config='experiments/siamfcpp/test/vot/siamfcpp_alexnet.yaml'   #VOT
if model==2:
    config='experiments/siamfcpp/test/otb/siamfcpp_alexnet-otb.yaml'#OTB  
if model==3:
    config='experiments/siamfcpp/test/dtb70/siamfcpp_alexnet-dtb70.yaml'#DTB
if model==4:
    config='experiments/siamfcpp/test/uav123/siamfcpp_alexnet-uav123.yaml'#UAV123
if model==5:
    config='experiments/siamfcpp/test/uav20l/siamfcpp_alexnet-uav20l.yaml'#UAV20L
if model==6:
    config='experiments/siamfcpp/test/uavdt/siamfcpp_alexnet-uavdt.yaml'#UAVDT
if model==7:
    config='experiments/siamfcpp/test/visdrone/siamfcpp_alexnet-visdrone.yaml'#visdrone

if __name__ == '__main__':
    # parsing
    #parser = make_parser()
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('-cfg', '--config', default=config,type=str,help='experiment configuration')
    parsed_args = parser.parse_args()

    # experiment config  #abspath 绝对路径
    exp_cfg_path = os.path.realpath(parsed_args.config)
    root_cfg.merge_from_file(exp_cfg_path)
    logger.info("Load experiment configuration at: %s" % exp_cfg_path)

    # resolve config  ROOT_PATH='/home/ubuntu/pytorch/pytorch-tracking/SiamFC++'
    root_cfg = complete_path_wt_root_in_cfg(root_cfg, ROOT_PATH) #把cfg中的相对路径,变成了绝对路径
    #root_cfg['task_name']
    #root_cfg['test']
    #root_cfg['train']
    root_cfg = root_cfg.test #获取test的yaml root_cfg['test']
    #root_cfg['track']
    #root_cfg['vos']
    task, task_cfg = specify_task(root_cfg) #获取任务 track or vos ,
    task_cfg.freeze()

    torch.multiprocessing.set_start_method('spawn', force=True)
    
    # build_siamfcpp_tester
    model = model_builder.build("track", task_cfg.model)
    # build pipeline
    pipeline = pipeline_builder.build("track", task_cfg.pipeline, model)#配置超参数
    # build tester
    testers = tester_builder("track", task_cfg.tester, "tester", pipeline)
    
    for tester in testers:
        tester.test()
