from __future__ import absolute_import

import os

import sys
sys.path.append(os.getcwd())

from got10k.experiments import *

from dasiamrpn import SiamRPNTracker
import argparse
import multiprocessing

multiprocessing.set_start_method('spawn',True)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='dasiamrpn tracking')
    #parser.add_argument('--dataset',default='VOT2018', type=str,help='datasets')
    #parser.add_argument('--save_path', default='./results', type=str,help='config file')
    #parser.add_argument('--snapshot', default=snapshot, type=str,help='snapshot of models to eval')
    parser.add_argument('--model_path', default='./models/dasiamrpn_1.pth', type=str, help='eval one special video')
    # parser.add_argument('--video', default='', type=str, help='eval one special video')
    #parser.add_argument('--vis', action='store_true',help='whether visualzie result')
    args = parser.parse_args()
   
    tracker = SiamRPNTracker(args.model_path)

    # 单机多卡 16b   dasiamrpn_23.pth -prec_score:0.789 -succ_score:0.595 -succ_rate:0.759
    #               dasiamrpn_5.pth -prec_score:0.797 -succ_score:0.577 -succ_rate:0.725

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
 
   # e.run(tracker,visualize=False)
    
    prec_score,succ_score,succ_rate=e.report([tracker.name])

    ss='-prec_score:%.3f -succ_score:%.3f -succ_rate:%.3f' % (float(prec_score),float(succ_score),float(succ_rate))
    
    print(args.model_path.split('/')[-1],ss)
