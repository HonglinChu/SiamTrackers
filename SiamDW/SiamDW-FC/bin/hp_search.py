from __future__ import absolute_import

import os
from got10k.experiments import *

from siamfc import SiamFCTracker
import argparse
import multiprocessing
multiprocessing.set_start_method('spawn',True)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from siamfc.utils import get_logger 
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
   
    # best instance_size=271  lr_box=0.25  penalty_k=0.20  window_influence=0.40 效果最好 0.814(1) 0.603(1)
    
    parser.add_argument('--model_path', default='./models/siamfcres22_30.pth', type=str, help='eval one special video')
    
    parser.add_argument('--scale_lr',default='0.59', type=parse_range,help='lr_box')#      
    parser.add_argument('--window_influence', default='0.126', type=parse_range,help='config file') 
    parser.add_argument('--scale_penalty', default='0.9745', type=parse_range,help='snapshot of models to eval')
    parser.add_argument('--instance_size', default='255', type=parse_range_int, help='eval one special video')
    #parser.add_argument('--vis', action='store_true',help='whether visualzie result')
    
    args = parser.parse_args()
    # 默认
    cfg = {'scale_lr': 0.59, 'window_influence': 0.126, 'scale_penalty': 0.9745, 'instance_size': 255} # 0.65
   
    logger.info('start training!')
    logger.info(args.model_path)
    
    Sum=len(args.instance_size)*len(args.scale_lr)*len(args.scale_penalty)*len(args.window_influence)
    count=0

    max_prec,max_succ=0,0
    index_prec,index_succ=1,1
    
    for i in args.instance_size:
        cfg['instance_size']=i

        for l in args.scale_lr:
            cfg['scale_lr']=l

            for k in args.scale_penalty:
                cfg['scale_penalty']=k 

                for w in args.window_influence:
                    cfg['window_influence']=w

                    count+=1

                    tracker = SiamFCTracker(args.model_path,True,0,cfg)
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
                    