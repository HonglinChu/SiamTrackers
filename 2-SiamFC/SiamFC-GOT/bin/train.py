from __future__ import absolute_import #??

import os

from got10k.datasets import * 

from siamfc import TrackerSiamFC 

import multiprocessing

multiprocessing.set_start_method('spawn',True)

if __name__ == '__main__':

    root_dir = os.path.abspath('data/GOT-10k')#获取当前工作目录

    seqs = GOT10k(root_dir, subset='train', return_meta=True)

    tracker = TrackerSiamFC(net_path=None) #优化器，GPU，损失函数，网络模型

    tracker.train_over(seqs)
