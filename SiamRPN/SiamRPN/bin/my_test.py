from __future__ import absolute_import

import os

import sys
import torch 
sys.path.append(os.getcwd())

from got10k.experiments import *

from siamrpn import SiamRPNTracker
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='siamrpn tracking')
    #parser.add_argument('--dataset',default='VOT2018', type=str,help='datasets')
    #parser.add_argument('--save_path', default='./results', type=str,help='config file')
    #parser.add_argument('--snapshot', default=snapshot, type=str,help='snapshot of models to eval')
    parser.add_argument('--model_path', default='./models/siamrpn_16-16b-128x1000.pth', type=str, help='eval one special video')
    # parser.add_argument('--video', default='', type=str, help='eval one special video')
    #parser.add_argument('--vis', action='store_true',help='whether visualzie result')
    args = parser.parse_args()
   
    tracker = SiamRPNTracker(args.model_path)

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

    # root_dir = os.path.abspath('datasets/VOT2018')           # VOT测试在评估阶段报错
    # e = ExperimentVOT(root_dir,version=2018,read_image=True, experiments=('supervised', 'unsupervised'))

    # root_dir = os.path.abspath('datasets/TColor128')
    # e = ExperimentTColor128(root_dir)

    # root_dir = os.path.abspath('datasets/Nfs')
    # e = ExperimentNfS(root_dir)

    # root_dir = os.path.abspath('datasets/LaSOT')
    # e = ExperimentLaSOT(root_dir)
 
    e.run(tracker,visualize=False)
    
    prec_score,succ_score,succ_rate=e.report([tracker.name])

    ss='-prec_score:%.3f -succ_score:%.3f -succ_rate:%.3f' % (float(prec_score),float(succ_score),float(succ_rate))
    
    print(args.model_path.split('/')[-1],ss)
