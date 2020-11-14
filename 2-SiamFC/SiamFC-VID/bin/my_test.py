from __future__ import absolute_import

import os
import sys
sys.path.append(os.getcwd())

from got10k.experiments import *

from siamfc import SiamFCTracker

# import multiprocessing
# multiprocessing.set_start_method('spawn',True)

model_path='./models/siamfc_30.pth'

# model_gpu=nn.DataParallel(model,device_ids=[0,1])# 多GPU并行计算

# output=model_gpu(input)

if __name__ == '__main__':   

    tracker = SiamFCTracker(model_path) #初始化一个追踪器
    
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

    # root_dir = os.path.abspath('datasets/VOT2018')
    # e = ExperimentVOT(root_dir,version=2018)

    # root_dir = os.path.abspath('datasets/TColor128')
    # e = ExperimentTColor128(root_dir)

    # root_dir = os.path.abspath('datasets/Nfs')
    # e = ExperimentNfS(root_dir)

    # root_dir = os.path.abspath('datasets/LaSOT')
    # e = ExperimentLaSOT(root_dir)

    e.run(tracker,visualize=False)#默认不开启可视化

    prec_score,succ_score,succ_rate=e.report([tracker.name])
    
    ss='-prec_score:%.3f -succ_score:%.3f -succ_rate:%.3f' % (float(prec_score),float(succ_score),float(succ_rate))
    
    print(args.model_path.split('/')[-1],ss)
