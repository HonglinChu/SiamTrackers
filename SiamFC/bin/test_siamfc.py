from __future__ import absolute_import

import os
from got10k.experiments import *

from siamfc import SiamFCTracker

import multiprocessing
multiprocessing.set_start_method('spawn',True)

gpu_id=0

model_path='./models/siamfc_30.pth'

# model_gpu=nn.DataParallel(model,device_ids=[0,1])# 多GPU并行计算

# output=model_gpu(input)

if __name__ == '__main__':

    tracker = SiamFCTracker(model_path,gpu_id) #初始化一个追踪器
    
    # root_dir = os.path.abspath('datasets/OTB')
    # e = ExperimentOTB(root_dir, version=2013)

    # root_dir = os.path.abspath('datasets/OTB')
    # e = ExperimentOTB(root_dir, version=2015)

    root_dir = os.path.abspath('datasets/UAV123')
    e = ExperimentUAV123(root_dir, version='UAV123')

    # root_dir = os.path.abspath('datasets/UAV123')
    # e = ExperimentUAV123(root_dir, version='UAV20L')

    # root_dir = os.path.abspath('datasets/DTB70')
    # e = ExperimentDTB70(root_dir)

    # root_dir = os.path.abspath('datasets/VOT2018')
    # e = ExperimentVOT(root_dir,version=2018)

    # root_dir = os.path.abspath('datasets/TColor128')
    # e = ExperimentTColor128(root_dir)

    # root_dir = os.path.abspath('datasets/Nfs')
    # e = ExperimentNfS(root_dir)

    # root_dir = os.path.abspath('datasets/LaSOT')
    # e = ExperimentLaSOT(root_dir)

    e.run(tracker,visualize=True) #run(tracker, visualize=False)

    e.report([tracker.name])
