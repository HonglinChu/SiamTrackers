# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np
import sys 
import time 
sys.path.append(os.path.abspath('.'))


import argparse
import os

import cv2
import torch
import numpy as np
import math
import sys
import shutil
from tqdm import tqdm 
sys.path.append(os.getcwd()) 

from siamcar.core.config import cfg
from siamcar.tracker.siamcar_tracker import SiamCARTracker
from siamcar.utils.bbox import get_axis_aligned_bbox
from siamcar.utils.model_load import load_pretrain
from siamcar.models.model_builder import ModelBuilder

from toolkit.utils.region import vot_overlap, vot_float2str
from bin.my_eval import evaluate

from toolkit.datasets import DatasetFactory 

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#-----------------------------------------SiamCar-Test------------------------------------------------------#
parser = argparse.ArgumentParser(description='siamcar tracking')

parser.add_argument('--dataset', '-d', default='VOT2018',type=str,help='datasets')

parser.add_argument('--tracker_name', '-t', default='siamcar',type=str,help='tracker  name')

parser.add_argument('--config', type=str, default='./models/siamcar_alexnet/config.yaml',help='config file')

parser.add_argument('--snapshot', default='./models/snapshot/checkpoint_e20.pth', type=str,  help='snapshot 1 models to eval')

parser.add_argument('--save_path', default='./results', type=str,help='snapshot of models to eval')

parser.add_argument('--video', default='', type=str,help='eval one special video') 

parser.add_argument('--vis', action='store_true',default=False, help='whether visualzie result')

args = parser.parse_args()

torch.set_num_threads(1)

def main():
    
    # choose to use gpu or cpu
    use_gpu=False 

    # load config
    cfg.merge_from_file(args.config)
    
    # create model
    model = ModelBuilder() 

    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')

    # create model
    model = ModelBuilder()

    model = load_pretrain(model, args.snapshot)
    
    model.eval().to(device)

    x = torch.randn(1, 3, 255, 255)
    zf = torch.randn(1, 3, 127, 127)
    if torch.cuda.is_available() and use_gpu:
        # model = model.cuda()
        x = x.cuda()
        zf = zf.cuda()
    # oup = model(x, zf)

    T_w = 50  # warmup
    T_t = 1000  # test 
    with torch.no_grad():
        model.template(zf)
        for i in range(T_w):
            oup = model.track(x)
        t_s = time.time()
        for i in range(T_t):
            oup = model.track(x) 
        t_e = time.time()
        print('speed: %.2f FPS' % (T_t / (t_e - t_s)))

if __name__ == '__main__':
    main()
