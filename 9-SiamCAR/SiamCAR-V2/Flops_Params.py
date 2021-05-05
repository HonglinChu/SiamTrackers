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

parser.add_argument('--snapshot', default='./models/snapshot/checkpoint_e27.pth', type=str,  help='snapshot 1 models to eval')

parser.add_argument('--save_path', default='./results', type=str,help='snapshot of models to eval')

parser.add_argument('--video', default='', type=str,help='eval one special video')

parser.add_argument('--vis', action='store_true',default=False, help='whether visualzie result')

args = parser.parse_args()

torch.set_num_threads(1)

from thop import profile
from thop.utils import clever_format

def main():

    # load config
    cfg.merge_from_file(args.config)
    
    # create model
    model = ModelBuilder() 

    backbone=model.backbone

    head=model.car_head

    x = torch.randn(1, 3, 255, 255)
    zf = torch.randn(1, 3, 127, 127)
     
    inp_z = torch.randn(1,256,6,6)
    inp_x = torch.randn(1,256,22,22)

    model.template(zf)

    oup = model.track(x)

    macs, params = profile(model, inputs=(x,), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)

    # backbone
    macs, params = profile(backbone, inputs=(x,), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print('backbone macs is ', macs)
    print('backbone params is ', params)
    
    #head
    macs, params = profile(head, inputs=(inp_z,inp_x), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print('head macs is ', macs)
    print('head params is ', params)

if __name__ == '__main__':
    main()
