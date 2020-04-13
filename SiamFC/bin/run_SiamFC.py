import numpy as np
import time
from siamfc import SiamFCTracker, config
import cv2
import glob
import os

# 这是OTB数据集的接口
def run_SiamFC(seq, rp, saveimage):
    x = seq.init_rect[0]
    y = seq.init_rect[1]
    w = seq.init_rect[2]
    h = seq.init_rect[3]

    tic = time.clock()
    # starting tracking
    tracker = SiamFCTracker(config.model_path, config.gpu_id)
    res = []
    for idx, frame in enumerate(seq.s_frames):
        frame = cv2.cvtColor(cv2.imread(frame), cv2.COLOR_BGR2RGB)
        if idx == 0:
            bbox = (x, y, w, h)
            tracker.init(frame, bbox)
            bbox = (bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]) # 1-idx
        else:
            bbox = tracker.update(frame)
        res.append((bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1])) # 1-idx
    duration = time.clock() - tic
    result = {}
    result['res'] = res
    result['type'] = 'rect'
    result['fps'] = round(seq.len / duration, 3)
    return result

