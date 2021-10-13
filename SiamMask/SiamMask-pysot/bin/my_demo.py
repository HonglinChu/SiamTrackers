from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import imageio
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from glob import glob
import sys 
sys.path.append(os.path.abspath('.'))
from siamrpnpp.core.config import cfg
from siamrpnpp.models.model_builder import ModelBuilder
from siamrpnpp.tracker.tracker_builder import build_tracker

torch.set_num_threads(1)

def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or  video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame

# def select_model(arg):

#     if arg==1:
#         config='models/siamrpnpp_alexnet/config.yaml'         #SiamRPNpp  -AlexNet   180fps
#         snapshot='models/siamrpnpp_alexnet/snapshot/checkpoint_e27-16.pth'
#     elif arg==2:
#         config='models/siammask_resnet/config.yaml'     #SiamRPN   -AlexNet   -OTB  180fps
#         snapshot='models/siammask_resnet/model.pth'
#     else:
#         print('no model is selected')
#         return 0

#     param='model_'+str(arg)
#     video='dataset/GOT-20/val/1/' #修改成你自己的测试视频路径（可以选择OTB视频测试）

#     return config, snapshot,video, param

# config,snapshot,video,param=select_model(2)  #选择模式1  SiamRPNpp 

config='models/siammask_resnet/config.yaml'
snapshot='models/siammask_resnet/model.pth'
param=''
video='datasets/GOT-20/val/1/'
def main():
    #load parameters
    parser = argparse.ArgumentParser(description='tracking demo')
    parser.add_argument('--config', type=str, help='config file',default=config)
    parser.add_argument('--snapshot', type=str, help='model name',default=snapshot)
    parser.add_argument('--video_name',type=str, help='videos or image files',default=video)
    args = parser.parse_args()

    # load config
    cfg.merge_from_file(args.config)
    
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()
    # load model
    model.load_state_dict(torch.load(args.snapshot,
        map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)
    
    first_frame = True
    
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    
    cv2.namedWindow('video', cv2.WND_PROP_FULLSCREEN)
    ################################变量初始化###################################
    sum = 0
    timer=0
    num=0
    gif_images=[]#gif图
    ############################################################################
    for frame in get_frames(args.video_name):
        start = cv2.getTickCount()
        #if num==0:#directory+imgname+".avi"
            #videoWriter = cv2.VideoWriter(directory+imgname+'.avi',cv2.VideoWriter_fourcc("X", "V", "I", "D"),50,(frame.shape[1],frame.shape[0]))#img.shape[1],img.shape[0]
        num=num+1
        if first_frame:
            try:
                init_rect = cv2.selectROI(video_name, frame, False, False)
            except:
                exit()
            
            tracker.init(frame, init_rect)
            first_frame = False
        else:
            outputs = tracker.track(frame)
            end = cv2.getTickCount()
            during = (end - start) / cv2.getTickFrequency()
            timer=timer+during
            if 'polygon' in outputs:
                polygon = np.array(outputs['polygon']).astype(np.int32)
                cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                              True, (0,255, 0), 3)
                mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                mask = mask.astype(np.uint8)
                mask = np.stack([mask, mask, mask*255]).transpose(1, 2, 0)
                gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                c = sorted(contours, key=cv2.contourArea, reverse=True)[0] #面积最大的轮廓区域
                rect_new2= cv2.boundingRect(c)
                frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
                cv2.rectangle(frame, (rect_new2[0], rect_new2[1]),
                               (rect_new2[0]+rect_new2[2], rect_new2[1]+rect_new2[3]),
                               (0, 0, 255), 2)
            else:
                bbox = list(map(int, outputs['bbox']))
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                              (0, 255, 0), 2)
                # f.write('\n'+str(bbox[0])+','+str(bbox[1])+','+str(bbox[2])+','+str(bbox[3]))
            cv2.putText(frame, imgname2, (5, 50), cv2.FONT_HERSHEY_COMPLEX, 2.0, (255, 0,0), 2)
            cv2.putText(frame, str(num), (5, 120), cv2.FONT_HERSHEY_COMPLEX, 2.0, (255, 0,0), 2)
            cv2.namedWindow(video_name,0)
            cv2.resizeWindow(video_name,1000,800)
            cv2.imshow(video_name, frame)
            #gif_images.append(frame)
            #videoWriter.write(frame)
            cv2.waitKey(30)
    #imageio.mimsave(directory+imgname+'.gif',gif_images,'GIF',duration = 0.02)#速度太慢
    #f.close()
    fps=int(num/timer)
    print('FPS:%d'%(fps))
    #videoWriter.release()
if __name__ == '__main__':
    main()
