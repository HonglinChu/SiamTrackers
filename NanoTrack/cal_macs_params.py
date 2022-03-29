
import argparse
import os

import torch
import sys 
sys.path.append(os.path.abspath('.'))

import argparse
import os

import torch 
import sys 
sys.path.append(os.getcwd())  

from nanotrack.core.config import cfg 

from nanotrack.models.model_builder import ModelBuilder 

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

from thop import profile 

from thop.utils import clever_format

def main():

    cfg.merge_from_file('./models/config/config.yaml')
    
    model = ModelBuilder() 

    x = torch.randn(1, 3, 255, 255)
    zf = torch.randn(1, 3, 127, 127) 

    model.template(zf)  
    
    macs, params = profile(model, inputs=(x,), verbose = False) 

    macs, params = clever_format([macs, params], "%.3f") 
    
    print('overall macs is ', macs) 
    
    print('overall params is ', params)

if __name__ == '__main__': 
    main()
