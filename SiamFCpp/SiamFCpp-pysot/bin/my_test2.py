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

#-----------------------------------------SiamCar-Test------------------------------------------------------#
parser = argparse.ArgumentParser(description='siamcar tracking')

parser.add_argument('--dataset', '-d', default='VOT2018',type=str,help='datasets')

parser.add_argument('--tracker_name', '-t', default='siamcar',type=str,help='tracker  name')

parser.add_argument('--config', type=str, default='./models/siamcar_alexnet/config.yaml',help='config file')

parser.add_argument('--snapshot', default='./models/snapshot/checkpoint_e20.pth', type=str,  help='snapshot 1 models to eval')

parser.add_argument('--save_path', default='./results2', type=str,help='snapshot of models to eval') #

parser.add_argument('--video', default='', type=str,help='eval one special video')

parser.add_argument('--vis', action='store_true',default=False, help='whether visualzie result')
# -----------------------------------------------------------------------------------------------------------#

# ------------------------------------------------UpdateNet--------------------------------------------------#
# parser.add_argument('--update_path', default='./updatenet/checkpoint_stage2-2/lr11_11/checkpoint2.pth.tar', type=str, help='the path of updatenet model')
# parser.add_argument('--update_stage', default=3, type=int, help='updatenet stage:0,1,2,3')
# parser.add_argument('--update_lr', default=0, type=int, help='updatenet stage:0,1,2,3')
# -----------------------------------------------------------------------------------------------------------#

#---------------------------------------------SiamCar-Eval--------------------------------------------#
parser.add_argument('--tracker_path', '-p', default='./results2', type=str,help='tracker result path')

parser.add_argument('--num', '-n', default=4, type=int,help='number of thread to eval')

parser.add_argument('--show_video_level', '-s', dest='show_video_level',action='store_true')

parser.set_defaults(show_video_level=False)
#------------------------------------------------------------------------------------------------------------#

args = parser.parse_args()

torch.set_num_threads(1)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main():
    # load config
    cfg.merge_from_file(args.config) 

    # -------------------------------------hp_search---------------------------------------#
    params = [0.0,0.0,0.0]
    
    # Interpolation learning rate
    params[0] =cfg.TRACK.LR 
    # Scale penalty 
    params[1]=cfg.TRACK.PENALTY_K
    # Window influence
    params[2] =cfg.TRACK.WINDOW_INFLUENCE 

    params_name = args.snapshot.split('/')[-1] + ' '+ args.dataset + '  lr-' + str(params[0]) + '  pk-' + '_' + str(params[1]) + '  win-' + '_' + str(params[2])

    # -------------------------------------hp_search---------------------------------------#
    
    # cur_dir = os.path.dirname(os.path.realpath(__file__))

    dataset_root = os.path.join('./datasets', args.dataset)
    
    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval() 
    
    # build tracker  siamos
    tracker = SiamCARTracker(model, cfg.TRACK)

    # create dataset 
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)
   
    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
        total_lost=0
        avg_speed =0  # linlin
        for v_idx, video in tqdm(enumerate(dataset)):
        #for v_idx, video in tqdm(dataset):
            if args.video != '':
                # test one special video 
                if video.name != args.video:
                    continue
            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = [] 
            for idx, (img, gt_bbox) in enumerate(video):
                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                    gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
                    gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]+gt_bbox[3]-1,
                    gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]]
                tic = cv2.getTickCount()
                if idx == frame_counter:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h] #[topx,topy,w,h]
                    tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)
                elif idx > frame_counter:
                    outputs = tracker.track(img)
                    pred_bbox = outputs['bbox']
                    if cfg.MASK.MASK:
                        pred_bbox = outputs['polygon']
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
            video_path = os.path.join(args.save_path, args.dataset, args.tracker_name,
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
            avg_speed += idx / toc

        print('Speed: {:3.1f}fps'.format(avg_speed/60))
        print(params_name)
        
        #print(" stage:{:d} model:{:s} epoch:{:s} update_lr:{:f}".format(args.update_stage,args.update_path, args.update_path.split('/')[-1],update_lr[args.update_lr]))
       
    else:
    # OPE tracking
        for v_idx, video in tqdm(enumerate(dataset)): 
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []
            for idx, (img, gt_bbox) in enumerate(video):
                tic = cv2.getTickCount()
                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h] #[topx,topy,w,h]
                    tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    scores.append(None)
                    if 'VOT2018-LT' == args.dataset: 
                        pred_bboxes.append([1]) 
                    else: 
                        pred_bboxes.append(pred_bbox)
                else: 
                    outputs = tracker.track(img)
                    pred_bbox = outputs['bbox']
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
                video_path = os.path.join(args.save_path, args.dataset, args.tracker_name,
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
                video_path = os.path.join(args.save_path, args.dataset, args.tracker_name, video.name)
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
                model_path = os.path.join(args.save_path, args.dataset, args.tracker_name)
                if not os.path.isdir(model_path):
                    os.makedirs(model_path)
                result_path = os.path.join(model_path, '{}.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
            # print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            #     v_idx+1, video.name, toc, idx / toc))
        print(params_name)
    # os.chdir(model_path)
    # save_file = '../%s' % dataset
    # shutil.make_archive(save_file, 'zip')
    #print('Records saved at', save_file + '.zip')
    evaluate(args)

if __name__ == '__main__':
    main()
