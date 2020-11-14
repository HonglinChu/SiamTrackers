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
from tqdm import tqdm
# from pysot.core.config import cfg
# from pysot.models.model_builder import ModelBuilder
# from pysot.tracker.tracker_builder import build_tracker
# from pysot.utils.bbox import get_axis_aligned_bbox
# from pysot.utils.model_load import load_pretrain

from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str
from dasiamrpn.tracker import SiamRPNTracker
from dasiamrpn.utils import get_axis_aligned_bbox

parser = argparse.ArgumentParser(description='dasiamrpn tracking')
parser.add_argument('--dataset',default='UAV123', type=str,help='datasets')
# parser.add_argument('--config', default=config, type=str,help='config file')
# parser.add_argument('--snapshot', default=snapshot, type=str,help='snapshot of models to eval')
parser.add_argument('--video', default='', type=str, help='eval one special video')
parser.add_argument('--vis', action='store_true',help='whether visualzie result')
args = parser.parse_args()

torch.set_num_threads(5)

#model_path='./models/SiamRPNBIG.model'

def main():

    # load config
    dataset_root='/home/ubuntu/pytorch/pytorch-tracking/DaSiamRPN/datasets/'+args.dataset
    #  tracker
    model_path ='./models/SiamRPNBIG.model'

    name='DaSiamRPN'

    gpu_id=0 #这里改成1不能正常运行

    tracker =  SiamRPNTracker(model_path,gpu_id)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)
    #算法的名字
    model_name = name

    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
        # restart tracking
        total_lost=0
        #for v_idx, video in enumerate(dataset):
        for video in tqdm(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = []
            state=dict()
            for idx, (img, gt_bbox) in enumerate(video):
               # print(idx)
                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                       gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]]
                tic = cv2.getTickCount()
                if idx == frame_counter:
                    
                    state=tracker.init(img, np.array(gt_bbox))#注意gt_bbox和gt_bbox_的区别
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox)) #1-based
                    pred_bbox = [cx-(w)/2, cy-(h)/2, w, h]#1-based
                    
                    pred_bboxes.append(1)

                elif idx > frame_counter:
                    state = tracker.update(img) 
                    pos=state['target_pos']
                    sz=state['target_sz']
                    pred_bbox=np.array([pos[0]-sz[0]/2, pos[1]-sz[1]/2, sz[0], sz[1]])
                    #pred_bbox=np.array([pos[0]+1-(sz[0]-1)/2, pos[1]+1-(sz[1]-1)/2, sz[0], sz[1]])

                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                    if overlap > 0:
                        # not lost
                        pred_bboxes.append(pred_bbox)
                    else:
                        # lost object
                        pred_bboxes.append(2)
                        frame_counter = idx + 5 # skip 5 frames
                        lost_number += 1
                else:
                    pred_bboxes.append(0)
                toc += cv2.getTickCount() - tic
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > frame_counter:
                    cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))],
                            True, (0, 255, 0), 3)
                    if cfg.MASK.MASK:
                        cv2.polylines(img, [np.array(pred_bbox, np.int).reshape((-1, 1, 2))],
                                True, (0, 255, 255), 3)
                    else:
                        bbox = list(map(int, pred_bbox))
                        cv2.rectangle(img, (bbox[0], bbox[1]),
                                      (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
            toc /= cv2.getTickFrequency()
            # save results
            video_path = os.path.join('results', args.dataset, model_name,
                    'baseline', video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x])+'\n')
            # print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
            #         v_idx+1, video.name, toc, idx / toc, lost_number))
            total_lost += lost_number
       # print("{:s} total lost: {:d}".format(model_name, total_lost))
    else:
        # OPE tracking    
        #for v_idx, video in enumerate(dataset):
        for video in tqdm(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []
            state=dict()
            for idx, (img, gt_bbox) in enumerate(video):
                tic = cv2.getTickCount()
                if idx == 0:
            
                    state=tracker.init(img, np.array(gt_bbox))#注意gt_bbox和gt_bbox_的区别
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    pred_bbox = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                   
                    scores.append(None)
                    if 'VOT2018-LT' == args.dataset:
                        pred_bboxes.append([1])
                    else:
                        pred_bboxes.append(pred_bbox)
                else:
                    state = tracker.update(img) 
                    pos=state['target_pos']
                    sz=state['target_sz']
                    pred_bbox=np.array([pos[0]-sz[0]/2, pos[1]-sz[1]/2, sz[0], sz[1]])
                    
                    pred_bboxes.append(pred_bbox)
                    #scores.append(outputs['best_score'])
                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > 0:
                    gt_bbox = list(map(int, gt_bbox))
                    pred_bbox = list(map(int, pred_bbox))
                    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                  (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                  (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
            toc /= cv2.getTickFrequency()
            # save results
            if 'VOT2018-LT' == args.dataset:
                video_path = os.path.join('results', args.dataset, model_name,
                        'longterm', video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path,
                        '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_001_confidence.value'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in scores:
                        f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            elif 'GOT-10k' == args.dataset:
                video_path = os.path.join('results', args.dataset, model_name, video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            else:
                model_path = os.path.join('results', args.dataset, model_name)
                if not os.path.isdir(model_path):
                    os.makedirs(model_path)
                result_path = os.path.join(model_path, '{}.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
            # print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            #     v_idx+1, video.name, toc, idx / toc))

if __name__ == '__main__':
    main()

