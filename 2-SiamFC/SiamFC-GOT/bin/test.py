from __future__ import absolute_import

import os
from got10k.experiments import *

from siamfc import TrackerSiamFC


if __name__ == '__main__':
    net_path = 'models/siamfc_alexnet_e50.pth'# 
    tracker = TrackerSiamFC(net_path=net_path) #初始化一个追踪器

    # root_dir = os.path.abspath('datasets/OTB')
    # e = ExperimentOTB(root_dir, version=2013)

    root_dir = os.path.abspath('datasets/OTB')
    e = ExperimentOTB(root_dir, version=2015)

    # root_dir = os.path.abspath('datasets/UAV123')
    # e = ExperimentUAV123(root_dir, version='UAV123')

    # root_dir = os.path.abspath('datasets/UAV123')
    # e = ExperimentUAV123(root_dir, version='UAV20L')

    # root_dir = os.path.abspath('datasets/DTB70')
    # e = ExperimentDTB70(root_dir)

    # root_dir = os.path.abspath('datasets/UAVDT')
    # e = ExperimentUAVDT(root_dir)

    # root_dir = os.path.abspath('datasets/VisDrone')
    # e = ExperimentVisDrone(root_dir)

    # root_dir = os.path.abspath('datasetssets/VOT2018')
    # e = ExperimentVOT(root_dir,version=2018)

    # root_dir = os.path.abspath('datasets/VOT2016')
    # e = ExperimentVOT(root_dir,version=2016)

    # root_dir = os.path.abspath('datasets/TColor128')
    # e = ExperimentTColor128(root_dir)

    # root_dir = os.path.abspath('datasets/Nfs')
    # e = ExperimentNfS(root_dir,fps=240) #高帧率

    #root_dir = os.path.abspath('datasets/LaSOT')
    #e = ExperimentLaSOT(root_dir)

    e.run(tracker,visualize=False)#默认不开启可视化

    e.report([tracker.name])
