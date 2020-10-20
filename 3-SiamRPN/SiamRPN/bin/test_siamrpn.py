from __future__ import absolute_import

import os
from got10k.experiments import *

from siamrpn import SiamRPNTracker

import multiprocessing
multiprocessing.set_start_method('spawn',True)

model_path='./models/siamrpn_45.pth'

#model_path='./models/siamrpn_35.pth' #38和35效果比较好

gpu_id=0

if __name__=='__main__':

#  for epoch in range(46,49):
#     print('epoch:%s'% epoch)
#     model_path = os.path.join('./data/models', 'siamrpn_%s.pth' % epoch)

    tracker = SiamRPNTracker(model_path,gpu_id)

    root_dir = os.path.abspath('datasets/OTB')
    e = ExperimentOTB(root_dir, version=2013)
    
    # root_dir = os.path.abspath('datasets/OTB')
    # e = ExperimentOTB(root_dir, version=2015)

    # root_dir = os.path.abspath('datasets/UAV123')
    # e = ExperimentUAV123(root_dir, version='UAV123')

    # root_dir = os.path.abspath('datasets/UAV123')
    # e = ExperimentUAV123(root_dir, version='UAV20L')

    # root_dir = os.path.abspath('datasets/DTB70')
    # e = ExperimentDTB70(root_dir)

    # root_dir = os.path.abspath('datasets/VOT2018') # VOT测试暂时遇到问题
    # e = ExperimentVOT(root_dir,version=2018,read_image=True, experiments=('supervised', 'unsupervised'))

    # root_dir = os.path.abspath('datasets/TColor128')
    # e = ExperimentTColor128(root_dir)

    # root_dir = os.path.abspath('datasets/Nfs')
    # e = ExperimentNfS(root_dir)

    # root_dir = os.path.abspath('datasets/LaSOT')
    # e = ExperimentLaSOT(root_dir)
 
    e.run(tracker,visualize=False)

    e.report([tracker.name])
