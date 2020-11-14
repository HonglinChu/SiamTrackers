from __future__ import absolute_import

import os
from got10k.experiments import *

from siamrpn import SiamRPNTracker
import argparse
import multiprocessing
multiprocessing.set_start_method('spawn',True)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from siamrpn.utils import get_logger 
import logging 
import numpy as np 
def parse_range(range_str):
    param = list(map(float, range_str.strip().split(',')))
    #return np.arange(*param)
    return np.array(param)

def parse_range_int(range_str):
    param = list(map(int, range_str.strip().split(',')))
    #return np.arange(*param)
    return np.array(param)

logger = get_logger('./models/logs/help_search.log')

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='siamrpn tracking')
   
    # best instance_size=271  lr_box=0.25  penalty_k=0.20  window_influence=0.40  0.814(1) 0.603(1)
    #      instance_size=271  lr_box=0.15  penalty_k=0.20  window_influence=0.40  0.819, 0.602
    #      instance_size=271  lr_box=0.15  penalty_k=0.15  window_influence=0.40  max_prec:0.820(2) max_succ:0.603(2)
    parser.add_argument('--model_path', default='./models/siamrpn_16-16b-128x1000.pth', type=str, help='eval one special video')
    
    parser.add_argument('--lr_box',default='0.15', type=parse_range,help='lr_box')#      
    parser.add_argument('--window_influence', default='0.40', type=parse_range,help='config file') 
    parser.add_argument('--penalty_k', default='0.12,0.14,0.14,0.18,0.22,0.24,0.25,0.3', type=parse_range,help='snapshot of models to eval')
    parser.add_argument('--instance_size', default='271', type=parse_range_int, help='eval one special video')
    #parser.add_argument('--vis', action='store_true',help='whether visualzie result')
    
    args = parser.parse_args()
    # 默认
    cfg = {'lr_box': 0.30, 'window_influence': 0.44, 'penalty_k': 0.22, 'instance_size': 271} # 0.65
   
    logger.info('start training!')
    
    logger.info(args.model_path)
    
    Sum=len(args.instance_size)*len(args.lr_box)*len(args.penalty_k)*len(args.window_influence)
    count=0

    max_prec,max_succ=0,0
    index_prec,index_succ=1,1
    
    for i in args.instance_size:
        cfg['instance_size']=i

        for l in args.lr_box:
            cfg['lr_box']=l

            for k in args.penalty_k:
                cfg['penalty_k']=k 

                for w in args.window_influence:
                    cfg['window_influence']=w

                    count+=1

                    tracker = SiamRPNTracker(args.model_path,cfg)
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
                    # print('model: %s  instance_size: %d   lr: %.3f   penalty_k: %.3f,  window_influence: %.2f' \
                    #     %(model_path.split('/')[-1],i,l,k,w))
                    logger.info('num:[{}/{}]  instance_size={:d}  lr_box={:.2f}  penalty_k={:.2f}  window_influence={:.2f}'.format(count,Sum,i,l,k,w))

                    prec_score,succ_score,succ_rate= e.report([tracker.name])

                    if float(prec_score)>max_prec:
                        max_prec,index_prec=prec_score,count
                    if float(succ_score)>max_succ:
                        max_succ,index_succ=succ_score,count
                    
                    ss='max_prec:%.3f(%d) max_succ:%.3f(%d) -prec_score:%.3f -succ_score:%.3f -succ_rate:%.3f' % \
                        (max_prec,index_prec,max_succ,index_succ,float(prec_score),float(succ_score),float(succ_rate))
                    logger.info(ss)
                    