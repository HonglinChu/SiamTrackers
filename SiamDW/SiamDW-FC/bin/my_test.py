from __future__ import absolute_import

import os
from got10k.experiments import *

from siamfc import SiamFCTracker

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='siamfcres22 tracking')
    #.add_argument('--dataset',default='VOT2018', type=str,help='datasets')
    #parser.add_argument('--save_path', default='./results', type=str,help='config file')
    #parser.add_argument('--snapshot', default=snapshot, type=str,help='snapshot of models to eval')
    parser.add_argument('--model_path', default='./models/siamfcres22_50.pth', type=str, help='eval one special video')
    # parser.add_argument('--video', default='', type=str, help='eval one special video')
    #parser.add_argument('--vis', action='store_true',help='whether visualzie result')
    args = parser.parse_args()

    tracker = SiamFCTracker(net_path=args.model_path,train=False) #初始化一个追踪器

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

    prec_score,succ_score,succ_rate=e.report([tracker.name])
    
    ss='-prec_score:%.3f -succ_score:%.3f -succ_rate:%.3f' % (float(prec_score),float(succ_score),float(succ_rate))
    
    print(args.model_path.split('/')[-1],ss)
