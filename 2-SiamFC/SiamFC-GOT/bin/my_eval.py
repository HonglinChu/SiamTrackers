from __future__ import absolute_import

import os

import sys
sys.path.append(os.getcwd())
from got10k.experiments import *

from siamfc import SiamFCTracker

import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='siamfc tracking')
    #.add_argument('--dataset',default='VOT2018', type=str,help='datasets')
    #parser.add_argument('--save_path', default='./results', type=str,help='config file')
    #parser.add_argument('--snapshot', default=snapshot, type=str,help='snapshot of models to eval')
    parser.add_argument('--model_path', default='./models/siamfc_41.pth', type=str, help='eval one special video')
    # parser.add_argument('--video', default='', type=str, help='eval one special video')
    #parser.add_argument('--vis', action='store_true',help='whether visualzie result')
    args = parser.parse_args()

    # OTB100 单机,单GPU,8b,siamfc_50.pth -prec_score:0.790 -succ_score:0.591 -succ_rate:0.735
    # OTB100 单机,2-GPU,8b,siamfc_50.pth -prec_score:0.756 -succ_score:0.563 -succ_rate:0.699

    # OTB100 单机,2-GPU,16b,siamfc_50.pth -prec_score:0.760 -succ_score:0.575 -succ_rate:0.721
    # OTB100 单机,2-GPU,16b,siamfc_38.pth -prec_score:0.755 -succ_score:0.569 -succ_rate:0.711

    tracker = SiamFCTracker(model_path=args.model_path) #初始化一个追踪器

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

    # e.run(tracker,visualize=False)#默认不开启可视化

    prec_score,succ_score,succ_rate=e.report([tracker.name])
    
    ss='-prec_score:%.3f -succ_score:%.3f -succ_rate:%.3f' % (float(prec_score),float(succ_score),float(succ_rate))
    
    print(args.model_path.split('/')[-1],ss)
