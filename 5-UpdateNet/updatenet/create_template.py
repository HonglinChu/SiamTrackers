
import sys
import cv2  # imread
import torch
import numpy as np
from os.path import realpath, dirname, join
import os
import pdb

from basenet import SiamRPNBIG,SiamRPNVOT
from net_upd import UpdateResNet512,UpdateResNet256
from tracker_upd import SiamRPN_init_upd,SiamRPN_track_upd
from utils_upd import get_axis_aligned_bbox, cxy_wh_2_rect, overlap_ratio #loss get_axis_aligned_rect function


# load SiamRPN network
#abspath=realpath(dirname(__file__))#获取当前文件的路径
abspath='./models/'
net_file = join(abspath, 'SiamRPNBIG.model') #输出512
net = SiamRPNBIG()
net.load_state_dict(torch.load(net_file))
net.eval().cuda()          #测试模式

step=2 #step=1,2,3
if step>1: #第一阶段是线性更新,第二阶段开始使用UpdateNet网络进行更新
    #load UpdateNet network
    updatenet = UpdateResNet512()
    #updatenet = UpdateResNet256() 

    update_model=torch.load('./models/vot2018.pth.tar')['state_dict']

    #update_model_fix = dict()
    #for i in update_model.keys():
    #    update_model_fix['.'.join(i.split('.')[1:])] = update_model[i]
    #updatenet.load_state_dict(update_model_fix)

    updatenet.load_state_dict(update_model)
    updatenet.eval().cuda()
else:
    updatenet=''

reset = 1; frame_max = 300
setfile = 'update_set'     #用于训练update网络的数据集
temp_path ='./updatenet/'+setfile+'_templates_step'+str(step)+'_std'  #step=1,2,3

if not os.path.isdir(temp_path):
    os.makedirs(temp_path)

video_path = './updatenet/data/lasot'
lists = open('./updatenet/data/'+setfile+'.txt','r')
list_file = [line.strip() for line in lists] #读取 update_set.txt 
category = os.listdir(video_path)#获取指定路径下的所有文件名字
category.sort()                  #排序

template_acc = []; template_cur = [];template_gt=[]

init0 = []; init = []; pre = []; gt = []  #init0 is reset init

for video in category:

    if video not in list_file:
        continue
    print(video) 

    gt_path = join(video_path,video, 'groundtruth.txt')

    ground_truth = np.loadtxt(gt_path, delimiter=',')

    num_frames = len(ground_truth);  # num_frames = min(num_frames, frame_max)

    img_path = join(video_path,video, 'img');

    imgFiles = [join(img_path,'%08d.jpg') % i for i in range(1,num_frames+1)]

    frame = 0

    while frame < num_frames:

        Polygon = ground_truth[frame] #x,y,w,h
        cx, cy, w, h = get_axis_aligned_bbox(Polygon)#zero-based

        if w*h!=0: # xywh2cxcywh不存在等于0的情况，polygon2cxcywh存在越界的情况

            image_file = imgFiles[frame]

            #2020-05-13
            
            target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
            im = cv2.imread(image_file)  # HxWxC

            state = SiamRPN_init_upd(im, target_pos, target_sz, net)  # init tracker
            #state = SiamRPN_init_upd(im, np.array(gt_bbox),net)  # init tracker

            template_acc.append(state['z_f'])#累积模板

            template_cur.append(state['z_f_cur'])#当前检测模板

            template_gt.append(state['gt_f_cur'])#当前帧gt框对应的特征图,这是我自己添加的源代码没有,确保这里是正确的

            init0.append(0); init.append(frame); frame_reset=0; pre.append(0);  gt.append(1)

            #初始化结束,开始跟踪
            while frame < (num_frames-1):

                frame = frame + 1; frame_reset=frame_reset+1

                image_file = imgFiles[frame]#获取下一张图片的标签

                if not image_file:
                    break 
                
                Polygon = ground_truth[frame] #x,y,w,h                
                cx, cy, w, h = get_axis_aligned_bbox(Polygon)#zero-based
                gt_pos, gt_sz = np.array([cx, cy]), np.array([w, h])
                state['gt_pos']=gt_pos
                state['gt_sz']=gt_sz

                im = cv2.imread(image_file)  #HxWxC
                #state  = SiamRPN_track(state, im)  # track
                
                state = SiamRPN_track_upd(state, im,updatenet)  # track

                #pdb.set_trace()

                template_acc.append(state['z_f'])#累积模板

                template_cur.append(state['z_f_cur'])#检测模板

                template_gt.append(state['gt_f_cur'])#当前帧gt框对应的特征图

                init0.append(frame_reset); init.append(frame); pre.append(1); 

                if frame==(num_frames-1): #last frame
                    gt.append(0)
                else:
                    gt.append(1)

                res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])#zero-based
                #计算当前检测的框和gt框的iou，如果不相交则丢失目标
                if reset:   #reset 默认是1                 
                    #gt_rect = get_axis_aligned_rect(ground_truth[frame])#x，y，w，h
                    rect=ground_truth[frame] #topx,topy,w,h
                    gt_rect=np.array([rect[0]-1,rect[1]-1,rect[2],rect[3]])#0-based x,y,w,h
                    iou = overlap_ratio(gt_rect, res)
                    if iou<=0:#这个条件是不是太宽了 iou<0.5才可以把
                        break    
        else:
            template_acc.append(torch.zeros([1, 512, 6, 6], dtype=torch.float32));  
            template_cur.append(torch.zeros([1, 512, 6, 6], dtype=torch.float32)); 
            template_gt.append(torch.zeros([1, 512, 6, 6], dtype=torch.float32))
            init0.append(0); init.append(frame); pre.append(1); 

            if frame==(num_frames-1): #last frame
                gt.append(0)
            else:
                gt.append(1)  

        frame = frame + 1 #skip

template_acc=np.concatenate(template_acc)#累积模板
template_cur=np.concatenate(template_cur)#当前检测模板
template_gt=np.concatenate(template_gt)#gt特征图

np.save(temp_path+'/template',template_acc) #累积特征图
np.save(temp_path+'/templatei',template_cur)#当前的检测模板
np.save(temp_path+'/template0',template_gt) #gt模板
np.save(temp_path+'/init0',init0)#第一帧  
np.save(temp_path+'/init',init)#累积模板编号   
np.save(temp_path+'/pre',pre)#每一帧对应的检测模板 =1 
np.save(temp_path+'/gt',gt)#每一帧对应的gt,一般=1